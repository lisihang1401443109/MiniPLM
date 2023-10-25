import random
import torch
import os
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size
from utils import print_rank
from tqdm import tqdm
import json
import numpy as np


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.pad_token_id
        self.eod_id = self.tokenizer.eos_token_id
        self.pad_id_in_data = self.tokenizer.pad_token_id
        assert self.eod_id != self.pad_id_in_data
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length
        self.answers = None
        self.order = None
        self.epoch = 0
        self.skip_offset = (-1, -1)
        if args.bin_data:
            self.data = DistributedMMapIndexedDataset(data_path, f"{split}", get_rank(), get_world_size(),
                                                      min_state=kwargs.get("min_state", 0), max_state=kwargs.get("max_state", None),
                                                      min_offset=kwargs.get("min_offset", 0), max_offset=kwargs.get("max_offset", None),
                                                      )
            
        elif args.json_data:
            self.data, self.origin_data = self.load_data_json(data_path)
        else:
            # txt data
            self.data = self.load_data_txt(data_path)
        
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            with open(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        elif os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            with open(os.path.join(data_path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
        
        if self.answers is not None:
            self.label_map = {tokenizer.encode(x[0], add_special_tokens=False)[0]: x[0] for x in self.answers}
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num PPO instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")
        else:
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        
        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt = d["prompt"].replace("<n>", "\n")
            prompt_ids = self.tokenizer.encode(prompt)
            output_ids = None
            if "output" in d:
                if isinstance(d["output"], list):
                    output_ids = self.tokenizer.encode(d["output"][0])
                else:
                    output_ids = self.tokenizer.encode(d["output"])
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids[:self.max_length - self.max_prompt_length]
            })
        print_rank("Load End")
        return data, data_origin

    def load_data_txt(self, data_path):
        with open(os.path.join(data_path, f"{self.split}.txt")) as f:
            lines = f.readlines()
        data = []
        print_rank("Loading Data")
        for line in lines:
            line = line.strip()
            line = line.replace("<n>", "\n")
            prompt = self.tokenizer.encode(line)
            data.append(prompt)
        print_rank("Load End")
        return data

    def verbalizer(self):
        return self.label_map

    def set_order(self, path):
        self.order = np.load(path, mmap_mode="r")
        assert self.order.shape[1] <= self.num
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_skip_offset(self, skip_offset):
        self.skip_offset = tuple(skip_offset)

    def __getitem__(self, index: int):
        if (self.epoch, index) < self.skip_offset:
            return None

        if self.order is not None:
            index = int(self.order[self.epoch, index])

        output_ids = None
        data = self.data[index]
        if self.args.bin_data:
            data = data.astype(int)
            if 65535 in data:
                source_len = np.where(data==65535)[0][0]
                output_ids = data[source_len+1:]
                data = data[:source_len]
            else:
                output_ids = data[1:]
                data = data[:1] # language modeling is the same as using the first token as prompt 
        elif self.args.json_data:
            output_ids = data["output_ids"]
            data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        prompt = data[:prompt_length]
        rest = data[prompt_length:]  

        if output_ids is not None:
            rest = output_ids  
    
        return index, prompt, rest

    def collate_lm(self, samples):
        
        if samples[0] is None:
            return None, None
        
        bs = len(samples)
        max_length = max([len(samp[1]) + len(samp[2]) for samp in samples])
        max_length = min(max_length-1, self.max_length)
        
        model_batch = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, self.max_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "loss_mask": torch.zeros(bs, max_length, dtype=torch.float)
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            full_ids = np.concatenate([prompt, rest], axis=0)[:max_length-1]
            model_batch["input_ids"][i][:len(full_ids)-1] = torch.tensor(full_ids[:-1], dtype=torch.long)
            model_batch["attention_mask"][i][:len(full_ids)-1] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["label"][i][:len(full_ids)-1] = torch.tensor(full_ids[1:], dtype=torch.long)
            st = max(len(prompt)-1, 0)
            no_model_batch["loss_mask"][i][:len(full_ids)-1] = (torch.tensor(full_ids[1:], dtype=torch.long) != self.pad_id_in_data)
            no_model_batch["loss_mask"][i][:st] = 0.0
        
        return model_batch, no_model_batch

    def collate_gen(self, samples):
        bs = len(samples)
        
        max_prompt_length = max([len(samp[1]) for samp in samples])
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)   
             
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch
