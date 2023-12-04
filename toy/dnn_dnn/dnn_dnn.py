import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os
import json
import time
from utils import print_rank, save_rank
from matplotlib import pyplot as plt

from torch.func import functional_call, vmap, grad

torch.autograd.set_grad_enabled(False)

class Model(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(Model, self).__init__()
        self.win = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.wout = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        x = self.win(x)
        x = self.act(x)
        x = self.wout(x)
        return x


class DNNDNN():
    def __init__(self, args, device, dim=None, real_dim=None, path=None):
        self.args = args
        self.device = device
        self.dim = dim
        self.real_dim = real_dim if real_dim is not None else dim
        self.params_gd = None
        self.train_data, self.dev_data = None, None
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.acc_rate_steps = [100, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        self.model = Model(self.dim, self.args.dnn_hidden_dim).to(self.device)
    
        self.compute_sample_grad = vmap(
            grad(self.loss_per_sample), in_dims=(None, 0, 0))
    
    def set_nn_gd(self):
        model_gd = Model(self.dim, self.args.gd_dnn_hidden_dim).to(self.device)
        self.params_gd = {k: v.detach().clone() for k, v in model_gd.named_parameters()}
        
    def generate_data(self, data_num, noise_scale, x_u, x_sigma, params_gd=None):
        x = torch.randn(data_num, self.dim, device=self.device) * x_sigma + x_u
        x[:, 0] = 1
        params_gd = self.params_gd if params_gd is None else params_gd
        y = functional_call(self.model, params_gd, x) + torch.randn(data_num, 1, device=self.device) * noise_scale
        return x, y
    
    def set_train_data(self, x, y):
        self.train_data = (x,y)
    
    def set_dev_data(self, x, y):
        self.dev_data = (x,y)

    def set_test_data(self, x, y):
        self.test_data = (x, y)
    
    def loss(self, params, x, y, alpha=None):
        pred = functional_call(self.model, params, x)
        if alpha is not None:
            loss = (alpha * ((pred - y)).pow(2)).sum()
        else:
            loss = (pred - y).pow(2).mean()
        return loss

    def loss_per_sample(self, params, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        pred = functional_call(self.model, params, x)
        loss = ((pred - y).pow(2)).sum()
        return loss

    def _train(self, alpha=None, state_init=None, IF_info=False, wandb_name="debug", epochs=None, log_interval=None):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        test_x, test_y = self.test_data
        
        epochs = self.args.epochs if epochs is None else epochs
        log_interval = self.args.log_interval if log_interval is None else log_interval
        
        run = wandb.init(
            name=f"{wandb_name}",
            project="toy-linear",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=["debug", self.args.time_stamp],)
        
        if state_init is not None:
            model = Model(self.dim, self.args.dnn_hidden_dim).to(self.device)
            model.load_state_dict(state_init)
        else:
            model = self.model
        
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        
        if alpha is None:
            alpha = torch.ones(self.args.train_num, 1, device=self.device)
            alpha = alpha / torch.sum(alpha)
        else:
            alpha = alpha.to(self.device)

        all_train_loss, all_dev_loss, all_test_loss = [], [], []
        for epoch in tqdm(range(epochs), desc=f"{wandb_name} Training"):
            loss = self.loss(params, train_x, train_y, alpha)
            loss_no_alpha = self.loss(params, train_x, train_y)
            
            g = grad(self.loss)(params, train_x, train_y, alpha)
            
            for name in g:
                g[name] += self.args.lam * params[name]
            
            dev_loss = self.loss(params, dev_x, dev_y)
            test_loss = self.loss(params, test_x, test_y)

            gn = 0
            for name in g:
                gn += torch.norm(g[name])

            for name, param in params.items():
                params[name] -= self.args.lr * g[name]
            
            wandb_log = {
                "train_loss": loss.item(),
                "train_loss_no_alpha": loss_no_alpha.item(),
                "dev_loss": dev_loss.item(),
                "test_loss": test_loss.item(),
                "grad_norm": gn
            }
            
            if IF_info:
                wandb_log.update({
                    "mean_IF": mean_IF.item(),
                    "var_IF": var_IF.item()
                })
            
            wandb.log(wandb_log)
            
            all_train_loss.append(loss.item())
            all_dev_loss.append(dev_loss.item())
            all_test_loss.append(test_loss.item())
            
            if (epoch % log_interval == 0):
                log_str = "Epoch: {} | Train Loss: {:.4f} | Train Loss No Alpha: {:.4f} | Dev Loss: {:.4f} | Test Loss: {:.4f} | Grad Norm: {:.4f}".format(
                    epoch, loss, loss_no_alpha, dev_loss, test_loss, gn)
                if IF_info:
                    log_str += " | Mean IF: {:.4f} | Var IF: {:.4f}".format(mean_IF, var_IF)
                print_rank(log_str)
                # save_rank(log_str, os.path.join(self.args.save, "log.txt"))

        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        
        dev_loss = self.loss(params, dev_x, dev_y)
        log_str = "Final Dev Loss: {}".format(dev_loss)
        print_rank(log_str)
        
        test_loss = self.loss(params, test_x, test_y)
        log_str = "Final Test Loss: {}".format(test_loss)
        print_rank(log_str)

        run.finish()

        return params, loss, dev_loss, all_train_loss, all_dev_loss, all_test_loss

    def train(self):
        self._train()

    def calc_acc_rate(self, baseline_losses, losses, steps=None):
        if steps is None:
            steps = self.acc_rate_steps
        
        def _calc(step):
            baseline_loss = baseline_losses[step]
            # binary search baseline_loss in losses
            l, r = 0, len(losses) - 1
            while l < r:
                mid = (l + r) // 2
                if losses[mid] >= baseline_loss:
                    l = mid + 1
                else:
                    r = mid
            return step / l
            
        acc_rate = [round(_calc(step), 3) for step in steps]

        return acc_rate
