import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
import numpy as np
import os
import sys
import wandb
import random
import time
import json
import math
from torch.func import grad, vmap, grad_and_value, functional_call
from utils import save_rank, print_rank

from toy.trm.tiny_story_model import ToyTSTransformer, ToyTokenizer
from toy.trm.base_trainer import ToyBaseTrainer
from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


class ToyTSTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(ToyTSTrainer, self).__init__(args, device)
        
        if args.ckpt_name in ["toy-trm", "toy-trm-ln", "toy-trm-rope", "toy-trm-silu", "toy-trm-silu-2"]:
            self.config = "toy"
        else:
            self.config = {
                "base_model_config": AutoConfig.from_pretrained(args.model_path)
            }
        self.tokenizer = ToyTokenizer(
            args.model_path, os.path.join(args.data_dir, "vocab.pt"))
        print_rank("vocab size: {}".format(self.tokenizer.vocab_size))
        self.max_length = args.max_length
        print_rank("max length: {}".format(self.max_length))

        self.model_init_dir = os.path.join(args.base_path, "processed_data", "toy-ts", "model_init")
        os.makedirs(self.model_init_dir, exist_ok=True)
        model_init_path = os.path.join(self.model_init_dir, f"{args.ckpt_name}.pt")
        
        self.model = self.get_model()
        print_rank(' > number of parameters: {}'.format(
            sum([p.nelement() for p in self.model.parameters()])), flush=True)
        
        if args.load_toy_data is not None:
            if not os.path.exists(model_init_path):
                torch.save(self.model.state_dict(), model_init_path)
            else:
                self.model.load_state_dict(torch.load(model_init_path))
        
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_iters)
        # self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
    
        self.train_data, self.dev_data, self.test_data = self.get_data()
        self.train_data = self.reform_data(self.train_data)
        self.dev_data = self.reform_data(self.dev_data)
        self.test_data = self.reform_data(self.test_data)
    
        print_rank("train data size: {} | dev data size: {} | test data size: {}".format(
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
        # labels[labels == self.tokenizer.pad_token_id] = -100
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
    
    def evaluate(self):
        dev_grad_acc_steps = self.dev_data[0].size(0) // self.args.eval_batch_size
        dev_losses, test_losses = [], []
        with torch.no_grad():
            for i in range(dev_grad_acc_steps):
                start = i * self.args.eval_batch_size
                end = (i+1) * self.args.eval_batch_size
                dev_loss = self.model.compute_loss(self.dev_data[0][start:end], self.dev_data[1][start:end])
                test_loss = self.model.compute_loss(self.test_data[0][start:end], self.test_data[1][start:end])
                dev_losses.append(dev_loss.item())
                test_losses.append(test_loss.item())
        dev_loss = np.mean(dev_losses)
        test_loss = np.mean(test_losses)
        
        return dev_loss, test_loss

    def train(self, alpha=None, wandb_name="baseline", calc_IF=False):
        save_path = os.path.join(self.args.save, wandb_name)
        os.makedirs(save_path, exist_ok=True)

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
        
        if self.args.batch_size == -1:
            self.args.batch_size = self.train_data[0].size(0)
        if self.args.eval_batch_size == -1:
            self.args.eval_batch_size = self.dev_data[0].size(0)
        
        assert self.train_data[0].size(0) % self.args.batch_size == 0, (self.train_data[0].size(0), self.args.batch_size)
        assert self.dev_data[0].size(0) % self.args.eval_batch_size == 0, (self.dev_data[0].size(0), self.args.eval_batch_size)
        
        min_dev_loss = 1e8
        min_dev_loss_epoch = -1
        
        if alpha is None:
            flat_alpha = torch.ones(self.train_data[0].size(0), device=self.device) / self.train_data[0].size(0)
        
        grad_acc_steps = self.train_data[0].size(0) // self.args.batch_size
        for e in range(self.args.epochs):
            epoch_st = time.time()
            self.optimizer.zero_grad()
            alpha_e = alpha[e] if alpha is not None else flat_alpha
            train_losses = []
            dev_loss, test_loss = self.evaluate()

            # params = {n: p.detach() for n, p in self.model.named_parameters()}
            # buffers = {n: p.detach() for n, p in self.model.named_buffers()}
            # g_params = {n:0 for n, _ in self.model.named_parameters()}
            
            for i in range(grad_acc_steps):
                if e == 0 and i == 0:
                    print(self.train_data[0][0])
                    print(self.tokenizer.decode(self.train_data[0][0].cpu().tolist()))
                    print()
                start = i * self.args.batch_size
                end = (i+1) * self.args.batch_size
                batch_alpha_e = alpha_e[start:end]
                
                loss = self.model.compute_loss(self.train_data[0][start:end], self.train_data[1][start:end], alpha=batch_alpha_e)
                loss.backward()
                
                # g, loss = grad_and_value(self.model.compute_loss_func)(params, buffers, self.model, self.train_data[0][start:end], self.train_data[1][start:end], alpha=batch_alpha_e)
                # for k in g_params.keys():
                #     g_params[k] += g[k]
                
                train_losses.append(loss.item())
            
            # for n, p in self.model.named_parameters():
            #     p.data.add_(g_params[n], alpha=-self.args.lr)

            loss = np.sum(train_losses)

            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            gn = self.get_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

            all_dev_loss.append(dev_loss)
            all_test_loss.append(test_loss)
            
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                min_dev_loss_epoch = e
                        
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
                log_str = "epoch {} | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | gn: {:.4f} | lr:{:.4e} | single epoch time: {}\n".format(
                    e, loss.item(), dev_loss.item(), test_loss.item(), gn, self.lr_scheduler.get_last_lr()[0], time.time() - epoch_st)
                if calc_IF:
                    log_str += "Dev IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        dev_IF_mean.item(), dev_IF_var.item(), dev_IF_std.item(), dev_IF_ratio.item())
                    log_str += "Test IF | IF_mean: {:.4f} | IF_var: {:.4f} | IF_std: {:.4f} | IF_ratio: {:.4f}\n".format(
                        test_IF_mean.item(), test_IF_var.item(), test_IF_std.item(), test_IF_ratio.item())
                print(log_str)
                save_rank(log_str, os.path.join(save_path, "log.txt"))
                # print(self.dev_data[0][:20], self.dev_data[1][:20], dev_preds[:20])
        
        print("all_time", time.time() - st)
        log_str = "min dev loss epoch: {} | min dev loss: {:.4f} | test loss: {:.4f}\n".format(min_dev_loss_epoch, min_dev_loss, all_test_loss[min_dev_loss_epoch])
        print(log_str)
        save_rank(log_str, os.path.join(save_path, "log.txt"))
        final_dev_loss, final_test_loss = self.evaluate()
          
        print("final | dev loss {:.4f} | test loss {:.4f}".format(final_dev_loss.item(), final_test_loss.item()))
        
        torch.save((all_dev_loss, all_test_loss), os.path.join(save_path, "all_loss.pt"))
        if calc_IF:
            torch.save((all_dev_IF, all_dev_IF_mean, all_dev_IF_var, all_dev_IF_std, all_dev_IF_ratio), os.path.join(save_path, "all_dev_IF.pt"))
            torch.save((all_test_IF, all_test_IF_mean, all_test_IF_var, all_test_IF_std, all_test_IF_ratio), os.path.join(save_path, "all_test_IF.pt"))
        
        run.finish()
            