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
from utils import save_rank, print_rank, all_gather
import torch.distributed as dist
from collections import defaultdict

from toy.trm.tiny_story_model import ToyTSTransformer, ToyTokenizer
from toy.trm.base_trainer import ToyBaseTrainer
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup)

from torch.optim.lr_scheduler import LinearLR


class ToyTSTrainer(ToyBaseTrainer):
    def __init__(self, args, device) -> None:
        super(ToyTSTrainer, self).__init__(args, device)
    
    def get_tokenizer(self):
        tokenizer = ToyTokenizer(
            self.args.model_path, os.path.join(self.args.data_dir, "vocab.pt"))
        return tokenizer
        
    def get_model(self):
        model = ToyTSTransformer(self.args, self.config).to(self.device)
        for p in model.parameters():
            dist.broadcast(p, 0)
        return model
        
    def get_data(self):
        all_data_splits = {}
        for split in ["dev", "test"]:
            data = torch.load(os.path.join(self.args.data_dir, f"{split}.pt"))
            all_data_splits[split] = data
        if self.args.add_noise is not None:
            all_data_splits["train"] = torch.load(os.path.join(self.args.data_dir, f"noise_train_{self.args.add_noise}.pt"))
        else:
            all_data_splits["train"] = torch.load(os.path.join(self.args.data_dir, "train.pt"))
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

    def calc_avg_mean_IF(self, e, dev_xn, dev_yn, test_xn, test_yn):
        grad_dev_vec = self.calc_grad_eval(dev_xn, dev_yn)
        grad_test_vec = self.calc_grad_eval(test_xn, test_yn)
        self.acmlt_grad_dev_vec += grad_dev_vec
        self.acmlt_grad_test_vec += grad_test_vec

        if (e+1) % self.args.avg_IF_calc_interval == 0:
            curr_param_vec = self.model.params_to_vector({n: p.detach() for n, p in self.model.named_parameters()})
            delta_theta = -(curr_param_vec - self.prev_param_vec)
            self.avg_mean_IF_dev = (self.acmlt_grad_dev_vec / self.args.avg_IF_calc_interval) @ delta_theta
            self.avg_mean_IF_test = (self.acmlt_grad_test_vec / self.args.avg_IF_calc_interval) @ delta_theta
            self.prev_param_vec = curr_param_vec
            self.acmlt_grad_dev_vec = 0
            self.acmlt_grad_test_vec = 0
            
        return self.avg_mean_IF_dev, self.avg_mean_IF_test

    def calc_avg_IFs2(self, e, xn, yn, alpha):
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        buffers = {n: p.detach() for n, p in self.model.named_buffers()}

        r = dist.get_rank()

        grad_train_single_func = grad(self.model.compute_loss_func_single)
        grad_train_func = vmap(grad_train_single_func, in_dims=(None, None, None, 0, 0))

        grad_bs = self.args.grad_batch_size
        gl_grad_bs = dist.get_world_size() * grad_bs
        grad_acc_steps = self.args.num_samp_grads // gl_grad_bs
        
        if self.sample_grads is None:
            self.sample_grads = [defaultdict(float) for _ in range(grad_acc_steps)]
        
        for i in range(grad_acc_steps):
            xn_batch = xn[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs]
            yn_batch = yn[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs]
            grad_train = grad_train_func(params, buffers, self.model, xn_batch, yn_batch)
    
            for n, _ in self.model.named_parameters():
                x2 = grad_train[n].contiguous().view(grad_train[n].size(0), -1)
                self.sample_grads[i][n] += x2

    def calc_avg_IFs(self, e, dev_xn, dev_yn, test_xn, test_yn, alpha):
        r = dist.get_rank()
        
        grad_dev_vec = self.calc_grad_eval(dev_xn, dev_yn)
        grad_test_vec = self.calc_grad_eval(test_xn, test_yn)
        self.acmlt_grad_dev_vec += grad_dev_vec
        self.acmlt_grad_test_vec += grad_test_vec

        grad_bs = self.args.grad_batch_size
        gl_grad_bs = dist.get_world_size() * grad_bs
        grad_acc_steps = self.args.num_samp_grads // gl_grad_bs

        if (e+1) % self.args.avg_IF_calc_interval == 0:
            curr_param_vec = self.model.params_to_vector({n: p.detach() for n, p in self.model.named_parameters()})
            delta_theta = -(curr_param_vec - self.prev_param_vec)
            
            tmp_grad_dev_vec = self.acmlt_grad_dev_vec / self.args.avg_IF_calc_interval
            tmp_grad_test_vec = self.acmlt_grad_test_vec / self.args.avg_IF_calc_interval
            self.avg_mean_IF_dev = tmp_grad_dev_vec @ delta_theta
            self.avg_mean_IF_test = tmp_grad_test_vec @ delta_theta
            self.prev_param_vec = curr_param_vec
            
            tmp_grad_dev = self.model.vector_to_params(tmp_grad_dev_vec)
            tmp_grad_test = self.model.vector_to_params(tmp_grad_test_vec)
            self.avg_IF_devs.zero_()
            self.avg_IF_tests.zero_()
            for i in range(grad_acc_steps):
                for n, _ in self.model.named_parameters():
                    self.avg_IF_devs[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs] += self.sample_grads[i][n] @ tmp_grad_dev[n].view(-1)
                    self.avg_IF_tests[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs] += self.sample_grads[i][n] @ tmp_grad_test[n].view(-1)
            self.sample_grads = None

            dist.all_reduce(self.avg_IF_devs, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.avg_IF_tests, op=dist.ReduceOp.SUM)
            
            self.acmlt_grad_dev_vec = 0
            self.acmlt_grad_test_vec = 0
            
        return self.avg_IF_devs, self.avg_IF_tests, self.avg_mean_IF_dev, self.avg_mean_IF_test