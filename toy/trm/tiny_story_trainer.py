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
import json
import math
from torch.func import grad, vmap

from toy.trm.tiny_story_model import ToyTSTransformer, ToyTokenizer
from toy.trm.base_trainer import ToyBaseTrainer
from transformers import AutoConfig, AutoTokenizer


class ToyTSTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(ToyTSTrainer, self).__init__(args, device)
        
        self.config = {
            "base_model_config": AutoConfig.from_pretrained(args.model_path)
        }
        self.tokenizer = ToyTokenizer(
            args.model_path, "/mnt/yuxian/data/tinystories/all_data/all_tokens_mistral.pt")
        print("vocab size: {}".format(self.tokenizer.vocab_size))
        self.max_length = args.max_length
        print("max length: {}".format(self.max_length))

        self.model_init_dir = os.path.join(args.base_path, "processed_data", "toy-ts", "model_init")
        os.makedirs(self.model_init_dir, exist_ok=True)
        model_init_path = os.path.join(self.model_init_dir, f"{args.ckpt_name}.pt")
        
        self.model = self.get_model()
        print(' > number of parameters: {}'.format(
            sum([p.nelement() for p in self.model.parameters()])), flush=True)
        
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
        self.model.load_state_dict(torch.load(os.path.join(self.model_init_dir, f"{args.model_name}.pt")))
    
    def get_model(self):
        return ToyTSTransformer(self.config).to(self.device)
        
    def get_data(self):
        all_data_splits = {}
        for split in ["train", "dev", "test"]:
            data = torch.load(os.path.join(self.args.data_dir, f"{split}.pt"))
            all_data_splits[split] = data
        all_data_splits["train"] = all_data_splits["train"][:self.args.train_num]
        all_data_splits["dev"] = all_data_splits["dev"][:self.args.dev_num]
        all_data_splits["test"] = all_data_splits["test"][:self.args.dev_num]
        return all_data_splits["train"], all_data_splits["dev"], all_data_splits["test"]
    
    def reform_data(self, data):
        assert data.size(1) == self.max_length + 1
        input_ids = data[:, :-1].clone().to(self.device)
        labels = data[:, 1:].clone().to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
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
        assert self.train_data[0].size(0) % self.args.batch_size == 0
        assert self.dev_data[0].size(0) % self.args.batch_size == 0
        grad_acc_steps = self.train_data[0].size(0) // self.args.batch_size
        dev_grad_acc_steps = self.dev_data[0].size(0) // self.args.batch_size
        
        for e in range(self.args.epochs):
            self.optimizer.zero_grad()
            alpha_e = alpha[e] if alpha is not None else None
            train_losses = []
            for i in range(grad_acc_steps):
                start = i * self.args.batch_size
                end = (i+1) * self.args.batch_size
                batch_alpha_e = alpha_e[start:end] if alpha is not None else None
                loss = self.model.compute_loss(self.train_data[0][start:end], self.train_data[1][start:end], alpha=batch_alpha_e)
                train_losses.append(loss.item())
                if alpha is None:
                    loss = loss / grad_acc_steps
                loss.backward()
            
            loss = np.mean(train_losses)

            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            gn = self.get_grad_norm()
            self.optimizer.step()
            dev_losses, test_losses = [], []
            with torch.no_grad():
                for i in range(dev_grad_acc_steps):
                    start = i * self.args.batch_size
                    end = (i+1) * self.args.batch_size
                    dev_loss = self.model.compute_loss(self.dev_data[0][start:end], self.dev_data[1][start:end])
                    test_loss = self.model.compute_loss(self.test_data[0][start:end], self.test_data[1][start:end])
                    dev_losses.append(dev_loss.item())
                    test_losses.append(test_loss.item())
                dev_loss = np.mean(dev_losses)
                test_loss = np.mean(test_losses)

            all_dev_loss.append(dev_loss)
            all_test_loss.append(test_loss)
            
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
                log_str = "epoch {} | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | gn: {:.4f}\n".format(
                    e, loss.item(), dev_loss.item(), test_loss.item(), gn)
                if calc_IF:
                    log_str += "Dev IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        dev_IF_mean.item(), dev_IF_var.item(), dev_IF_std.item(), dev_IF_ratio.item())
                    log_str += "Test IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        test_IF_mean.item(), test_IF_var.item(), test_IF_std.item(), test_IF_ratio.item())
                print(log_str)
                # print(self.dev_data[0][:20], self.dev_data[1][:20], dev_preds[:20])
        
        print(time.time() - st)
        # final_loss = self.model.compute_loss(*self.train_data)
        # final_dev_loss, = self.model.compute_loss(*self.dev_data)
        # final_test_loss, = self.model.compute_loss(*self.test_data)
          
        # print("final | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f}".format(
        #     final_loss.item(), final_dev_loss.item(), final_test_loss.item()))
        
        save_path = os.path.join(self.args.save, wandb_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save((all_dev_loss, all_test_loss), os.path.join(save_path, "all_loss.pt"))
        if calc_IF:
            torch.save((all_dev_IF, all_dev_IF_mean, all_dev_IF_var, all_dev_IF_std, all_dev_IF_ratio), os.path.join(save_path, "all_dev_IF.pt"))
            torch.save((all_test_IF, all_test_IF_mean, all_test_IF_var, all_test_IF_std, all_test_IF_ratio), os.path.join(save_path, "all_test_IF.pt"))
        
        run.finish()
            