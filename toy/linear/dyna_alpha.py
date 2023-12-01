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

    def train_alpha_t(self):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        
        alpha = torch.ones(self.args.train_num, 1, device=self.device, requires_grad=True)
        alpha = alpha / torch.sum(alpha)
        
        assert self.theta_init is not None
        theta = torch.clone(self.theta_init)

        norm_vec = torch.ones(self.args.train_num, device=self.device)
        norm_vec = norm_vec / torch.norm(norm_vec)

        all_train_loss, all_dev_loss, all_test_loss = [], [], []

        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            loss = self.loss(train_x, train_y, theta, alpha)
            dev_loss = self.loss(dev_x, dev_y, theta)
            
            grad_full = 2 * alpha_norm * train_x * (train_x @ theta - train_y) # (train_num, dim)
            grad = torch.sum(grad_full, dim=0).unsqueeze(-1)  + self.args.lam * theta # (dim, 1)
            grad_dev = 2 / self.args.dev_num * dev_x.t() @ (dev_x @ theta - dev_y) # (dim, 1)
            IF = grad_full @ grad_dev # (train_num, 1)
            mean_IF = torch.mean(IF)
            var_IF = torch.var(IF)
            all_mean_IF.append(mean_IF.item())
            all_var_IF.append(var_IF.item())
            
            theta -= self.args.lr * grad
            
            all_train_loss.append(loss.item())
            all_dev_loss.append(dev_loss.item())
            
            var_IF.backward()
            print(alpha.grad)
            delta_alpha = alpha.grad - norm_vec * (torch.dot(norm_vec, alpha.grad.squeeze()))
            print(delta_alpha)
            print(torch.sum(delta_alpha))
            
            alpha -= self.args.lr_alpha * delta_alpha
            del alpha.grad
            
            exit(0)

            self.writer.add_scalar("train_loss", loss.item(), epoch)
            self.writer.add_scalar("dev_loss", dev_loss.item(), epoch)
            self.writer.add_scalar("mean_IF", mean_IF.item(), epoch)
            self.writer.add_scalar("var_IF", var_IF.item(), epoch)
            
            if epoch % 10 == 0:
                log_str = "Epoch: {} | Train Loss: {:.4f} | Dev Loss: {:.4f}".format(epoch, loss, dev_loss)
                print_rank(log_str)
            # save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                
        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        dev_loss = self.loss(dev_x, dev_y, theta)
        log_str = "Final Dev Loss: {}".format(dev_loss)
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        self.save_and_plot(all_train_loss, "train_loss")
        self.save_and_plot(all_dev_loss, "dev_loss")
        self.save_and_plot(all_mean_IF, "mean_IF")
        self.save_and_plot(all_var_IF, "var_IF")

