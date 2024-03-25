import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
from tqdm import tqdm
import os
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader, DistributedSampler
from utils import print_rank, get_model, save_rank, save_parallel, get_tokenizer, all_gather
from data_utils.prompt_datasets import PromptDataset

from train_eval_utils.base_trainer import BaseTrainer


class ContrastiveDecoder(BaseTrainer):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device, False)
        self.setup_model_and_optimizer(set_optim=False)
        self.setup_amateur_model() 

    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        data_split = self.args.data_split if self.args.data_split is not None else "test"
        self.eval_dataset = PromptDataset(args, self.tokenizer, data_split, args.data_dir, args.dev_num)

    def setup_amateur_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        assert args.base_model_path is not None
        base_model = get_model(args, device, args.base_model_path, from_scratch=False)
        base_model.eval()
        self.base_model = base_model
        
    def generate(self, batch):
        generation_config = self.get_generattion_config(batch)
        gen_out = self.model.generate(
            **generation_config,
            amateur_model=self.base_model,
            amateur_alpha = self.args.amateur_alpha,
            amateur_beta = self.args.amateur_beta,
        )
        return gen_out
    
    def _decode(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_gen)
        
        self.model.eval()
        all_prompt_ids, all_response_ids = [], []
        all_gen_times = []

        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"Generation Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                prompt_ids = model_batch["input_ids"]
                st = time()
                gen_out = self.generate(model_batch)
                gen_time = time() - st
                response_ids = gen_out["sequences"][:, prompt_ids.size(1):]
                all_prompt_ids.append(torch.nn.functional.pad(prompt_ids, (self.args.max_prompt_length-prompt_ids.size(1), 0), value=self.tokenizer.pad_token_id))
                all_response_ids.append(torch.nn.functional.pad(response_ids, (0, self.args.max_length-response_ids.size(1)), value=self.tokenizer.pad_token_id))
                all_gen_times.append(gen_time)
        
        all_prompt_ids = torch.cat(all_prompt_ids, dim=0)
        all_prompt_ids = all_gather(all_prompt_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_prompt_ids = all_prompt_ids.view(-1, all_prompt_ids.size(-1))
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))

        all_gen_times = all_gather(torch.tensor(all_gen_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
        gen_time = all_gen_times.sum().item()

        if get_rank() == 0:
            response_strs = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)            
            res = {"gen_time": gen_time}
        else:
            res, response_strs = None, None
        
        dist.barrier()
        return all_prompt_ids, all_response_ids, res, response_strs

    def decode(self):
        all_prompt_ids, all_response_ids, res, response_strs = self._decode()
        if get_rank() == 0:
            res_str = self.get_log(res, "decode")
            print_rank(res_str)
            save_rank(res_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
            torch.save({"prompt_ids": all_prompt_ids, "response_ids": all_response_ids}, os.path.join(self.args.save, "decode_results.pt"))
            with open(os.path.join(self.args.save, "decode_results.json"), "w") as f:
                json.dump(response_strs, f)
        dist.barrier()