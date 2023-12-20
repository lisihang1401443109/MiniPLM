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

from toy.trm.model import ToyTransformer


class ToyTrmTrainer():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        
        self.config = {
            "vocab_size": 12,
            "max_len": 6,
            "hidden_size": args.input_dim,
            "num_head": 4,
        }
        
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.data_dir = os.path.join(
            args.base_path, "processed_data", "toy-add", f"tn{args.train_num}-dn{args.dev_num}-r{args.ratio_1_2}", f"{args.seed}-{args.seed_data}")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.model = ToyTransformer(self.config).to(device)
        
        if args.load_toy_data is None:
            torch.save(self.model.state_dict(), os.path.join(self.data_dir, "model_init.pt"))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.data_dir, "model_init.pt")))
        
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)
    
        self.train_data, self.dev_data, self.test_data = self.get_data()
        self.train_data = self.reform_data(self.train_data)
        self.dev_data = self.reform_data(self.dev_data)
        self.test_data = self.reform_data(self.test_data)
    
        print("train data size: {} | dev data size: {} | test data size: {}".format(
            (self.train_data[0].size(), self.train_data[1].size()), 
            (self.dev_data[0].size(), self.dev_data[1].size()), 
            (self.test_data[0].size(), self.test_data[1].size())))
    
    def get_label(self, x, y):
        return ((x + y) // 10) % 10
    
    def generate_data(self):
        origin_state = random.getstate()
        random.seed(self.args.seed_data)
        all_data = []
        for i in range(100):
            all_data.extend([(i, j, self.get_label(i,j)) for j in range(100)])
        random.shuffle(all_data)
        dev_data = all_data[:self.args.dev_num]
        test_data = all_data[self.args.dev_num:2*self.args.dev_num]
        train_data = all_data[2*self.args.dev_num:]
        
        split_1 = [x for x in train_data if x[2] < 5]
        split_2 = [x for x in train_data if x[2] >= 5]
        
        ratio_1_2 = self.args.ratio_1_2
        if ratio_1_2 > 1:
            split_2 = split_2[:int(len(split_2) / ratio_1_2)]
        else:
            split_1 = split_1[:int(len(split_1) * ratio_1_2)]
            
        train_data = split_1 + split_2
        
        random.shuffle(train_data)
        train_data = train_data[:self.args.train_num]
        
        train_data = torch.tensor(train_data, dtype=torch.long)
        dev_data = torch.tensor(dev_data, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.long)

        torch.save((train_data, dev_data, test_data), os.path.join(self.data_dir, "data.pt"))
        
        random.setstate(origin_state)
        
        return (train_data, dev_data, test_data)

    def get_data(self):
        if self.args.load_toy_data is not None:
            data = torch.load(os.path.join(self.data_dir, "data.pt"))
        else:
            data = self.generate_data()
        
        return data
    
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
    
    def forward(self, batch):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        input_ids = batch[:, :-1]
        labels = batch[:, -1]
        logits = self.model(input_ids)
        logits = logits[:, -1, :]
        losses = loss_fn(logits, labels)
        loss = torch.mean(losses)
        
        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum(preds == labels).item() / labels.size(0)
        return loss, acc
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def train(self, wandb_name="baseline"):
        
        run = wandb.init(
            name=f"{wandb_name}",
            project="toy-trm",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=[self.args.time_stamp],)
        
        st = time.time()
        for e in range(self.args.epochs):
            self.optimizer.zero_grad()
            loss, _ = self.forward(self.train_data)
            loss.backward()
            gn = self.get_grad_norm()
            for p in self.model.parameters():
                # p.data = p.data - self.args.lr * p.grad.data
                p.data.add_(p.grad.data, alpha=-self.args.lr)
            # self.optimizer.step()
            dev_loss, dev_acc = self.forward(self.dev_data)
            test_loss, test_acc = self.forward(self.test_data)
            # print(e)
            # print("train loss", loss.item())
            # print(dev_loss.item())
            wandb_log = {
                "train_loss": loss.item(),
                "dev_loss": dev_loss.item(),
                "test_loss": test_loss.item(),
                "dev_acc": dev_acc,
                "test_acc": test_acc,
                "grad_norm": gn
            }
            
            wandb.log(wandb_log)
            
            if e % self.args.log_interval == 0:
                print("epoch {} | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | dev_acc: {:.4f} | test_acc: {:.4f} | gn: {:.4f}".format(
                    e, loss.item(), dev_loss.item(), test_loss.item(), dev_acc, test_acc, gn))
        print(time.time() - st)
        final_loss, _ = self.forward(self.train_data)
        final_dev_loss, final_dev_acc = self.forward(self.dev_data)
        final_test_loss, final_test_acc = self.forward(self.test_data)
          
        print("final | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | dev acc: {:.4f} | test acc: {:.4f}".format(
            final_loss.item(), final_dev_loss.item(), final_test_loss.item(), final_dev_acc, final_test_acc))
        
        run.finish()
            