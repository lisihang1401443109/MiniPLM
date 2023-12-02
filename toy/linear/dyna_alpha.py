import torch
import wandb
from tqdm import tqdm
import os
import json
import time
from utils import print_rank, save_rank
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from .linear_model import LinearModel


class LinearModelDynaAlpha(LinearModel):
    def __init__(self, args, device, dim=None, path=None):
        super(LinearModelDynaAlpha, self).__init__(args, device, dim, path)

    def train(self):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        test_x, test_y = self.test_data
        
        print("Baseline")
        
        baseline_out = self._train(wandb_name="baseline", IF_info=True)
        baseline_dev_losses = baseline_out[-2]
        baseline_test_losses = baseline_out[-1]

        run = wandb.init(
            name=f"dyna_alpha",
            project="toy-linear",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=["debug", self.args.time_stamp],)

        assert self.theta_init is not None
        theta = torch.clone(self.theta_init)

        alpha = torch.ones(self.args.train_num, 1, device=self.device, requires_grad=True)
        alpha = alpha / torch.sum(alpha)

        norm_vec = torch.ones(self.args.train_num, device=self.device)
        norm_vec = norm_vec / torch.norm(norm_vec)

        all_train_loss, all_dev_loss, all_test_loss = [], [], []

        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            loss = self.loss(train_x, train_y, theta, alpha)
            dev_loss = self.loss(dev_x, dev_y, theta)
            test_loss = self.loss(test_x, test_y, theta)
            
            grad_full_no_alpha = 2 * train_x * (train_x @ theta - train_y) # (train_num, dim)
            grad_dev = 2 / self.args.dev_num * dev_x.t() @ (dev_x @ theta - dev_y) # (dim, 1)
            IF = -grad_full_no_alpha @ grad_dev # (train_num, 1)
            mean_IF = torch.mean(IF / self.args.train_num)
            var_IF = torch.var(IF / self.args.train_num)

            delta_alpha = IF.squeeze() - norm_vec * (torch.dot(norm_vec, IF.squeeze()))
            delta_alpha = delta_alpha.unsqueeze(-1)

            alpha -= self.args.lr_alpha * delta_alpha

            alpha = torch.clamp(alpha, min=0)
            alpha = alpha / torch.sum(alpha)
            
            grad_full = alpha * grad_full_no_alpha # (train_num, dim)
            grad = torch.sum(grad_full, dim=0).unsqueeze(-1)  + self.args.lam * theta # (dim, 1)
            theta -= self.args.lr * grad
            
            wandb.log({
                "train_loss": loss.item(),
                "dev_loss": dev_loss.item(),
                "test_loss": test_loss.item(),
                "mean_IF": mean_IF.item(),
                "var_IF": var_IF.item()})

            all_train_loss.append(loss.item())
            all_dev_loss.append(dev_loss.item())
            all_test_loss.append(test_loss.item())

            if epoch % self.args.log_interval == 0:
                log_str = "Epoch: {} | Train Loss: {:.4f} | Dev Loss: {:.4f} | Test Loss: {:.4f} | Mean IF: {:.4f} | Var IF: {:.4f}".format(
                    epoch, loss, dev_loss, test_loss, mean_IF, var_IF)
                print_rank(log_str)

        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        
        dev_loss = self.loss(dev_x, dev_y, theta)
        log_str = "Final Dev Loss: {}".format(dev_loss)
        print_rank(log_str)
        
        test_loss = self.loss(test_x, test_y, theta)
        log_str = "Final Test Loss: {}".format(test_loss)
        print_rank(log_str)
        
        dev_acc_rate = self.calc_acc_rate(baseline_dev_losses, all_dev_loss)
        test_acc_rate = self.calc_acc_rate(baseline_test_losses, all_test_loss)
        
        log_str = f"Dev Acc Rate: {dev_acc_rate} | Test Acc Rate: {test_acc_rate}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        wandb.log({
            "dev_acc_rate": wandb.plot.line_series(
                xs=self.acc_rate_steps,
                ys=[dev_acc_rate]),
        })
        
        run.finish()
