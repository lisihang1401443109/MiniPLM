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

from addition_trainer import ToyAdditionTrainer
from tiny_story_trainer_dp import ToyTSTrainer
from logistic_trainer import LogisticTrainer


class EvalAlphaTrainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        if args.data_names == "addition":
            base_trainer_cls = ToyAdditionTrainer
        elif args.data_names == "tiny_story":
            base_trainer_cls = ToyTSTrainer
        elif args.data_names == "linear":
            base_trainer_cls = LogisticTrainer
        else:
            raise NotImplementedError(args.data_names)
        
        self.base_trainer = base_trainer_cls(args, device)
        
    def train(self, wandb_name=None):
        alpha_epochs = self.args.alpha_epochs.split(",")
        l = alpha_epochs[0]
        alpha_epochs = alpha_epochs[1:]
        if alpha_epochs[0] == "b":
            self.base_trainer.train(wandb_name="baseline", calc_IF=True)
            self.base_trainer.reload_model()
            alpha_epochs = alpha_epochs[1:]
        alpha_epochs = [int(epoch) for epoch in alpha_epochs]
        for alpha_epoch in alpha_epochs:
            alpha = torch.load(os.path.join(self.args.load_alpha, f"epoch_{alpha_epoch}", "opt_alpha.pt"))
            alpha = alpha.to(self.device)
            self.base_trainer.train(alpha=alpha, wandb_name="opt_alpha_{}/{}".format(l, alpha_epoch), calc_IF=True)
            self.base_trainer.reload_model()
