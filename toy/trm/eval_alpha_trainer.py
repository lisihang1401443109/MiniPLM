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

from trainer import ToyTrmTrainer


class EvalAlphaTrainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.base_trainer = ToyTrmTrainer(args, device)
        
    def train(self):
        self.base_trainer.train(wandb_name="baseline")
        self.base_trainer.reload_model()
        for alpha_epoch in [5, 10, 15, 20, 25, 30, 35, 40]:
            alpha = torch.load(os.path.join(self.args.load_alpha, f"epoch_{alpha_epoch}", "opt_alpha.pt"))
            self.base_trainer.train(alpha=alpha, wandb_name="opt_alpha_{}".format(alpha_epoch))
            self.base_trainer.reload_model()
