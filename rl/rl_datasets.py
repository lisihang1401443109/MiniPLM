from dataclasses import dataclass
from typing import Iterable
from torchtyping import TensorType
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from data_utils.prompt_datasets import PromptDataset
import torch
import torch.distributed as dist

try:
    from transformers import mpu
except:
    mpu = None

@dataclass
class ModelBatch:

    input_ids: TensorType["batch_size", "prompt_length"]
    attention_mask: TensorType["batch_size", "prompt_length"]


class RLPromptDataset(PromptDataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__(args, tokenizer, split, data_path, num)
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = max([len(samp[1]) for samp in samples])
        
        model_batch = ModelBatch(
            input_ids=torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            attention_mask=torch.zeros(bs, max_prompt_length, dtype=torch.long),
        )
        
        for i, (_, prompt, _) in enumerate(samples):
            # left padding
            model_batch.input_ids[i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch.attention_mask[i][-len(prompt):] = 1
        
        return model_batch
            
    def create_dataloader(self, shuffle, drop_last, batch_size, num_workers):
        if self.args.model_parallel:
            dp_world_size = mpu.get_data_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
        else:
            dp_world_size = dist.get_world_size()
            dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )
    
    def move_to_device(self, model_batch: ModelBatch, device):
        for k in model_batch.__dict__:
            model_batch.__dict__[k] = model_batch.__dict__[k].to(device)
        return model_batch

