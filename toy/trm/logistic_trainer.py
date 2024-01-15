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

from toy.trm.logistic_model import LogisticModel
from toy.trm.base_trainer import ToyBaseTrainer


class LogisticTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(LogisticTrainer, self).__init__(args, device)

    def get_tokenizer(self):
        return None
    
    def get_model(self):
        config = {
            "hidden_size": self.args.input_dim
        }
        model = LogisticModel(config).to(self.device)
        return model

    def get_data(self):
        train_data = torch.load(os.path.join(self.args.data_dir, "train.pt"))
        dev_data = torch.load(os.path.join(self.args.data_dir, "dev.pt"))
        test_data = torch.load(os.path.join(self.args.data_dir, "test.pt"))

        return train_data, dev_data, test_data

    def reform_data(self, data):
        return data[0].to(self.device), data[1].to(self.device)