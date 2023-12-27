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

from toy.trm.model import ToyTransformer
from toy.trm.base_trainer import ToyBaseTrainer


class ToyAdditionTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(ToyAdditionTrainer, self).__init__(args, device)
        
        self.config = {
            "vocab_size": 12,
            "max_len": 6,
            "hidden_size": args.input_dim,
            "num_head": args.num_head,
        }
        
        self.data_dir = os.path.join(
            args.base_path, "processed_data", "toy-add", f"tn{args.train_num}-dn{args.dev_num}-r{args.ratio_1_2}", f"{args.seed}-{args.seed_data}")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.model = self.get_model()
        
        model_init_path = os.path.join(self.data_dir, f"model_init_{args.input_dim}_{args.num_head}.pt")
        if args.load_toy_data is None or not os.path.exists(model_init_path):
            torch.save(self.model.state_dict(), model_init_path)
        else:
            self.model.load_state_dict(torch.load(model_init_path))
        
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)
    
        self.train_data, self.dev_data, self.test_data = self.get_data()
        self.train_data = self.reform_data(self.train_data)
        self.dev_data = self.reform_data(self.dev_data)
        self.test_data = self.reform_data(self.test_data)
    
        print("train data size: {} | dev data size: {} | test data size: {}".format(
            (self.train_data[0].size(), self.train_data[1].size()), 
            (self.dev_data[0].size(), self.dev_data[1].size()), 
            (self.test_data[0].size(), self.test_data[1].size())))
    
    def reload_model(self):
        self.model.load_state_dict(torch.load(os.path.join(
            self.data_dir, f"model_init_{self.args.input_dim}_{self.args.num_head}.pt")))
    
    def get_model(self):
        return ToyTransformer(self.config).to(self.device)
    
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
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def calc_IF(self, xn, yn, eval_xn, eval_yn, alpha=None):
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        buffers = {n: p.detach() for n, p in self.model.named_buffers()}
        grad_eval_func = grad(self.model.compute_loss_func)
        grad_eval = grad_eval_func(params, buffers, self.model, eval_xn, eval_yn)
        
        grad_train_single_func = grad(self.model.compute_loss_func_single)
        grad_train_func = vmap(grad_train_single_func, in_dims=(None, None, None, 0, 0))
        grad_train = grad_train_func(params, buffers, self.model, xn, yn)
        
        IF = torch.zeros(xn.size(0), device=self.device)
        for n, _ in self.model.named_parameters():
            x1 = grad_eval[n].view(-1)
            x2 = grad_train[n].contiguous().view(grad_train[n].size(0), -1)
            IF += x2 @ x1
            
        if alpha is None:
            IF_mean = torch.mean(IF, dim=0)
        else:
            IF_mean = torch.sum(alpha * IF, dim=0)
        
        IF_var = torch.var(IF, dim=0)
        IF_std = torch.std(IF, dim=0)
        IF_ratio = IF_mean / (IF_std + 1e-8)
        
        return IF, IF_mean, IF_var, IF_std, IF_ratio
    
    def train(self, alpha=None, wandb_name="baseline", calc_IF=False):
        
        run = wandb.init(
            name=f"{wandb_name}",
            project="toy-trm",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=[self.args.time_stamp],)
        
        st = time.time()
        all_dev_loss, all_test_loss = [], []
        all_dev_IF_mean, all_dev_IF_var, all_dev_IF_std, all_dev_IF_ratio = [], [], [], []
        all_test_IF_mean, all_test_IF_var, all_test_IF_std, all_test_IF_ratio = [], [], [], []
        all_dev_IF, all_test_IF = [], []
        for e in range(self.args.epochs):
            self.optimizer.zero_grad()
            alpha_e = alpha[e] if alpha is not None else None
            loss, _, _ = self.model.compute_loss(*self.train_data, alpha=alpha_e)
            loss.backward()
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            gn = self.get_grad_norm()
            self.optimizer.step()
            dev_loss, dev_acc, dev_preds = self.model.compute_loss(*self.dev_data)
            test_loss, test_acc, test_preds = self.model.compute_loss(*self.test_data)
            
            all_dev_loss.append(dev_loss.item())
            all_test_loss.append(test_loss.item())
            
            if calc_IF:
                dev_IF, dev_IF_mean, dev_IF_var, dev_IF_std, dev_IF_ratio = self.calc_IF(*self.train_data, *self.dev_data, alpha=alpha_e)
                all_dev_IF_mean.append(dev_IF_mean.item())
                all_dev_IF_var.append(dev_IF_var.item())
                all_dev_IF_std.append(dev_IF_std.item())
                all_dev_IF_ratio.append(dev_IF_ratio.item())
                all_dev_IF.append(dev_IF)
                
                test_IF, test_IF_mean, test_IF_var, test_IF_std, test_IF_ratio = self.calc_IF(*self.train_data, *self.test_data, alpha=alpha_e)
                all_test_IF_mean.append(test_IF_mean.item())
                all_test_IF_var.append(test_IF_var.item())
                all_test_IF_std.append(test_IF_std.item())
                all_test_IF_ratio.append(test_IF_ratio.item())
                all_test_IF.append(test_IF)
            
            wandb_log = {
                "train_loss": loss.item(),
                "dev_loss": dev_loss.item(),
                "test_loss": test_loss.item(),
                "dev_acc": dev_acc,
                "test_acc": test_acc,
                "grad_norm": gn,

            }
            if calc_IF:
                wandb_log.update({
                    "dev_IF_mean": dev_IF_mean.item(),
                    "dev_IF_var": dev_IF_var.item(),
                    "dev_IF_std": dev_IF_std.item(),
                    "dev_IF_ratio": dev_IF_ratio.item(),
                })
            
            wandb.log(wandb_log)
            
            if e % self.args.log_interval == 0:
                log_str = "epoch {} | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | dev_acc: {:.4f} | test_acc: {:.4f} | gn: {:.4f}\n".format(
                    e, loss.item(), dev_loss.item(), test_loss.item(), dev_acc, test_acc, gn)
                if calc_IF:
                    log_str += "Dev IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        dev_IF_mean.item(), dev_IF_var.item(), dev_IF_std.item(), dev_IF_ratio.item())
                    log_str += "Test IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        test_IF_mean.item(), test_IF_var.item(), test_IF_std.item(), test_IF_ratio.item())
                print(log_str)
                # print(self.dev_data[0][:20], self.dev_data[1][:20], dev_preds[:20])
        
        print(time.time() - st)
        final_loss, _, _ = self.model.compute_loss(*self.train_data)
        final_dev_loss, final_dev_acc,  _ = self.model.compute_loss(*self.dev_data)
        final_test_loss, final_test_acc, _ = self.model.compute_loss(*self.test_data)
          
        print("final | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | dev acc: {:.4f} | test acc: {:.4f}".format(
            final_loss.item(), final_dev_loss.item(), final_test_loss.item(), final_dev_acc, final_test_acc))
        
        save_path = os.path.join(self.args.save, wandb_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save((all_dev_loss, all_test_loss), os.path.join(save_path, "all_loss.pt"))
        if calc_IF:
            torch.save((all_dev_IF, all_dev_IF_mean, all_dev_IF_var, all_dev_IF_std, all_dev_IF_ratio), os.path.join(save_path, "all_dev_IF.pt"))
            torch.save((all_test_IF, all_test_IF_mean, all_test_IF_var, all_test_IF_std, all_test_IF_ratio), os.path.join(save_path, "all_test_IF.pt"))
        
        run.finish()
            