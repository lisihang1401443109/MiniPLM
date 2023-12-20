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


class LogisticTrainer():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device

        self.config = {
            "hidden_size": 128
        }

        self.exp_name = args.save.strip(
            "/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        # self.data_dir = os.path.join(
        #     args.base_path, "processed_data", "toy-add", f"tn{args.train_num}-dn{args.dev_num}-r{args.ratio_1_2}", f"{args.seed}-{args.seed_data}")
        # os.makedirs(self.data_dir, exist_ok=True)

        self.model = LogisticModel(self.config).to(device)

        if args.load_toy_data is None:
            raise NotImplementedError
        else:
            d = torch.load(os.path.join(self.args.load_toy_data, "data.pt"))
            theta_init = d[-1]
            self.model.load_state_dict({"linear.weight": theta_init.view(1, self.config["hidden_size"])})

        self.optimizer = SGD(self.model.parameters(), lr=args.lr)

        xn, yn, dev_xn, dev_yn, test_xn, test_yn = self.get_data()
        self.train_data = (xn, yn)
        self.dev_data = (dev_xn, dev_yn)
        self.test_data = (test_xn, test_yn)

        print("train data size: {} | dev data size: {} | test data size: {}".format(
            (self.train_data[0].size(), self.train_data[1].size()),
            (self.dev_data[0].size(), self.dev_data[1].size()),
            (self.test_data[0].size(), self.test_data[1].size())))

    def get_data(self):
        if self.args.load_toy_data is not None:
            data = torch.load(os.path.join(self.args.load_toy_data, "data.pt"))
        else:
            raise NotImplementedError

        return data[:-1]

