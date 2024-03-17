import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
from arguments import get_args

from utils import print_args, initialize
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer

from kd.trainer import KDTrainer
from sft.trainer import SFTTrainer
from mos.trainer import MOSKDTrainer
from mos.trainer import MOSSFTTrainer
from pretrain.trainer import PreTrainer
from pretrain.residual_trainer import ResidualPreTrainer
from pretrain.residual_kd_trainer import ResidualKDPreTrainer
from pretrain.kd_trainer import KDPreTrainer
from pretrain.contrastive_kd_trainer import ContrastiveKDPreTrainer


torch.set_num_threads(16)


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.deepspeed_config = None
    
    if args.type == "kd":
        trainer = KDTrainer(args, ds_config, device, args.do_train)
    elif args.type == "sft":
        trainer = SFTTrainer(args, ds_config, device, args.do_train)
    elif args.type == "mos_kd":
        trainer = MOSKDTrainer(args, ds_config, device, args.do_train)
    elif args.type == "mos_sft":
        trainer = MOSSFTTrainer(args, ds_config, device, args.do_train)
    elif args.type == "pretrain":
        trainer = PreTrainer(args, ds_config, device, args.do_train)
    elif args.type == "pt_rsd":
        trainer = ResidualPreTrainer(args, ds_config, device, args.do_train)
    elif args.type == "kd_rsd":
        trainer = ResidualKDPreTrainer(args, ds_config, device, args.do_train)
    elif args.type == "kd_pretrain":
        trainer = KDPreTrainer(args, ds_config, device, args.do_train)
    elif args.type == "kd_contrastive":
        trainer = ContrastiveKDPreTrainer(args, ds_config, device, args.do_train)
    else:
        raise NotImplementedError        
    
    trainer.train()

    
if __name__ == "__main__":
    main()
