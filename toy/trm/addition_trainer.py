import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import os
import sys
import wandb
import random
import time
from torch.func import grad, vmap
import torch.distributed as dist

from toy.trm.addition_model import ToyAddTransformer, ToyTokenizer
from toy.trm.base_trainer import ToyBaseTrainer

from transformers import AutoConfig


class ToyAddTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(ToyAddTrainer, self).__init__(args, device)

    def get_tokenizer(self):
        tokenizer = ToyTokenizer()
        return tokenizer
    
    def get_model(self):
        config = {
            "base_model_config": AutoConfig.from_pretrained(self.args.model_path) if self.args.model_path is not None else None,
            "toy": "toy-trm" in self.args.ckpt_name
        }
        
        model = ToyAddTransformer(self.args, config).to(self.device)
        for p in model.parameters():
            dist.broadcast(p, 0)
        return model
    
    def reform_data(self, data):
        new_data = []
        for x in data:
            d1 = [int(p) for p in "{:0=2d}".format(x[0])]
            d2 = [int(p) for p in "{:0=2d}".format(x[1])]
            d3 = [int(x[2])]
            d = d1 + [10] + d2 + [11] + d3 # 10: +, 11: =
            new_data.append(d)
        
        new_data = torch.tensor(new_data, dtype=torch.long, device=self.device)
        return new_data[:, :-1], new_data[:, -1]
