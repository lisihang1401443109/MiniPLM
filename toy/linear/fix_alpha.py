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


class LinearModelFixAlpha(LinearModel):
    def __init__(self, args, device, dim=None, path=None):
        super(LinearModelFixAlpha, self).__init__(args, device, dim, path)
    
    def train(self):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        test_x, test_y = self.test_data
        
        alpha = torch.ones(self.args.train_num, 1, device=self.device)
        alpha = alpha / torch.sum(alpha)
        
        norm_vec = torch.ones(self.args.train_num, device=self.device)
        norm_vec = norm_vec / torch.norm(norm_vec)

        best_alpha = None
        best_outer_epoch = None
        best_dev_loss = float("inf")

        ood_init_theta = self.generate_rand_theta()

        train_outputs = self._train(alpha=alpha, theta_init=ood_init_theta, wandb_name=f"eval-init")
        init_test_losses = train_outputs[-1]

        for outer_epoch in range(self.args.outer_epochs):

            theta, loss, dev_loss, _, _, _ = self._train(
                alpha=alpha,
                theta_init=self.theta_init,
                wandb_name=f"oe-{outer_epoch}")

            grad_dev = 2 / self.args.dev_num * dev_x.t() @ (dev_x @ theta - dev_y) # (dim, 1)
            grad_train_full = 2 * train_x * (train_x @ theta - train_y) # (train_num, dim)
            grad_train_full = grad_train_full + self.args.lam * theta.squeeze().unsqueeze(0) # (train_num, dim)
            H_full = train_x.unsqueeze(-1) @ train_x.unsqueeze(-2) + self.args.lam * torch.eye(self.dim, device=self.device).unsqueeze(0) # (train_num, dim, dim)
            inv_H_full = torch.inverse(H_full)
            grad_alpha = -(grad_train_full.unsqueeze(-2) @ inv_H_full @ grad_dev).squeeze() # (train_num)
            proj_grad_alpha = grad_alpha - norm_vec * (torch.dot(norm_vec, grad_alpha))
            proj_grad_alpha = proj_grad_alpha.unsqueeze(-1)
            alpha -= self.args.lr_alpha * proj_grad_alpha
            
            alpha = torch.clamp(alpha, min=0)
            alpha = alpha / torch.sum(alpha)
            
            train_outputs = self._train(alpha=alpha, theta_init=ood_init_theta, wandb_name=f"eval-oe-{outer_epoch}")
            acc_rate = self.calc_acc_rate(init_test_losses, train_outputs[-1])
            naive_alpha = (alpha > 1e-10).float()
            naive_alpha = naive_alpha / torch.sum(naive_alpha)
            train_outputs = self._train(alpha=naive_alpha, theta_init=ood_init_theta, wandb_name=f"eval-naive-oe-{outer_epoch}")
            acc_rate_naive = self.calc_acc_rate(init_test_losses, train_outputs[-1])
            
            log_str = f"Outer Epoch: {outer_epoch} | Acc Rate: {acc_rate} | Acc Rate Naive: {acc_rate_naive}"
            print_rank(log_str)
            save_rank(log_str, os.path.join(self.args.save, "log.txt"))
            
            if dev_loss < best_dev_loss:
                best_alpha = alpha.clone()
                best_dev_loss = dev_loss
                best_outer_epoch = outer_epoch
        
        print_rank("##### Final Evaluate #####")
        
        print_rank(f"Best Dev Loss: {best_dev_loss}")
        self._train(alpha=best_alpha, theta_init=ood_init_theta, wandb_name=f"eval-best-oe-{best_outer_epoch}")
        torch.save(best_alpha, os.path.join(self.args.save, "best_alpha.pt"))

        naive_best_alpha = (best_alpha > 1e-10).float()
        naive_best_alpha = naive_best_alpha / torch.sum(naive_best_alpha)
        self._train(alpha=naive_best_alpha, theta_init=ood_init_theta, wandb_name=f"eval-naive-best-oe-{best_outer_epoch}")

        torch.save(naive_best_alpha, os.path.join(self.args.save, "naive_best_alpha.pt"))

        return best_alpha, best_outer_epoch, best_dev_loss