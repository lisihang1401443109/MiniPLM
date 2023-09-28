import torch
import os
from utils import get_rank
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torchtyping import TensorType
from dataclasses import dataclass
from typing import Iterable
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Element:
    prompt_ids: TensorType["prompt_length"]
    response_ids: TensorType["response_length"]


@dataclass
class Batch:
    prompt_ids: TensorType["batch_size", "prompt_length"]
    response_ids: TensorType["batch_size", "response_length"]
    


class RolloutStorage(Dataset):
    """
    Rollout storage
    """

    def __init__(self, args, tokenizer, seed):
        super().__init__()

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.rollouts = [None]
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def push(self, exps):
        self.rollouts += exps

    def save(self, path):
        def exp_to_dict(exp):
            return {k: v for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.rollouts]
        
        torch.save(data, os.path.join(path, f"rollouts_{get_rank()}.pkl"))
            
    def load(self, path):
        data = torch.load(os.path.join(path, f"rollouts_{get_rank()}.pkl"), map_location="cpu")
        self.rollouts = [Element(**d) for d in data]

    def clear_history(self):
        self.rollouts = []

    def trim(self, ids):
        trimmed_ids = ids[ids != self.pad_token_id]
        if ids[-1] == self.eos_token_id and trimmed_ids[-1] != self.eos_token_id:
            trimmed_ids = torch.cat([trimmed_ids, torch.tensor([self.eos_token_id])], dim=0)
        if ids[0] == self.bos_token_id and trimmed_ids[0] != self.bos_token_id:
            trimmed_ids = torch.cat([torch.tensor([self.bos_token_id]), trimmed_ids], dim=0)
        return trimmed_ids

    def __getitem__(self, index: int):
        elem = self.rollouts[index]
        elem.prompt_ids = self.trim(elem.prompt_ids)
        elem.response_ids = self.trim(elem.response_ids)
        return elem        

    def __len__(self) -> int:
        return len(self.rollouts)

    def collate(self, elems: Iterable[Element]):
        if any([e is None for e in elems]):
            print(elems)
        return Batch(
            pad_sequence(
                [elem.prompt_ids.flip(0) for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            ).flip(1), # left pad
            pad_sequence(
                [elem.response_ids for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            ),
        )

    def create_dataloader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        # sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last)
        # we don't use distributed sampler because the dataset on each device is different
        return DataLoader(
            self, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, generator=self.rng
        )
        
    def broadcast(self, batch, src=0, group=None):
        for k, v in batch.__dict__.items():
            dist.broadcast(batch.__dict__[k], src=src, group=group)
            
    def move_to_device(self, batch: Batch, device):
        for k, v in batch.__dict__.items():
            batch.__dict__[k] = batch.__dict__[k].to(device)