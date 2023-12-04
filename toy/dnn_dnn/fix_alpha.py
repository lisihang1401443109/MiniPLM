import torch
import wandb
from tqdm import tqdm
import os
import json
import time
from utils import print_rank, save_rank
from matplotlib import pyplot as plt
from dnn_dnn import DNNDNN, Model
from torch.func import hessian


class DNNDNNFixAlpha(DNNDNN):
    def __init__(self, args, device, dim=None, real_dim=None, path=None):
        super(DNNDNNFixAlpha, self).__init__(args, device, dim, real_dim, path)
        # self.compute_hessian = vmap(hessian(self.loss_per_sample_alpha), in_dims=(None, 0, 0, 0))
    
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

        ood_state_init = Model(self.dim, self.args.dnn_hidden_dim).to(self.device).state_dict()

        train_outputs = self._train(alpha=alpha, state_init=ood_state_init, wandb_name=f"eval-init")
        init_test_losses = train_outputs[-1]

        for outer_epoch in range(self.args.outer_epochs):

            params, loss, dev_loss, _, _, _ = self._train(
                alpha=alpha,
                state_init=self.model.state_dict(),
                wandb_name=f"oe-{outer_epoch}")

            grad_dev = self.compute_sample_grad(params, dev_x, dev_y)
            for name, grad in grad_dev.items():
                grad_dev[name] = torch.mean(grad_dev[name], dim=0).view(-1)
            grad_train_full = self.compute_sample_grad(params, train_x, train_y)
            for name in grad_train_full:
                grad_train_full[name] = grad_train_full[name] + self.args.lam * params[name].unsqueeze(0)
                grad_train_full[name] = grad_train_full[name].view(grad_train_full[name].shape[0], -1)

            grad_alpha = 0
            for name in grad_train_full:
                grad_alpha += -torch.sum(grad_train_full[name] * grad_dev[name], dim=1)
            
            # H_full = self.compute_hessian(params, train_x, train_y, train_avg_alpha)
            # for name in H_full:
            #     H_full[name] = H_full[name] + self.args.lam * torch.eye(self.dim, device=self.device).unsqueeze(0)
            #     inv_H_full = torch.inverse(H_full[name])
            #     grad_alpha = -(grad_train_full[name].unsqueeze(-2) @ inv_H_full @ grad_dev[name]).squeeze() # (train_num)
            proj_grad_alpha = grad_alpha - norm_vec * (torch.dot(norm_vec, grad_alpha))
            proj_grad_alpha = proj_grad_alpha.unsqueeze(-1)

            # plot alpha bar
            origin_alpha_plot = torch.sort(alpha.squeeze(), descending=True)[0].cpu().numpy()            
            plt.bar(range(self.args.train_num), origin_alpha_plot)
            plt.savefig(os.path.join(self.args.save, f"origin-alpha-{outer_epoch}.png"))
            plt.close()
            
            alpha -= self.args.lr_alpha * proj_grad_alpha
            
            alpha = torch.clamp(alpha, min=0)
            alpha = alpha / torch.sum(alpha)
            
            # plot alpha bar
            alpha_plot = torch.sort(alpha.squeeze(), descending=True)[0].cpu().numpy()            
            plt.bar(range(self.args.train_num), alpha_plot)
            plt.savefig(os.path.join(self.args.save, f"alpha-{outer_epoch}.png"))
            plt.close()
                   
            train_outputs = self._train(alpha=alpha, state_init=ood_state_init, wandb_name=f"eval-oe-{outer_epoch}")
            acc_rate = self.calc_acc_rate(init_test_losses, train_outputs[-1])
            
            naive_alpha = (alpha > 1e-10).float()
            naive_alpha = naive_alpha / torch.sum(naive_alpha)
            
            # plot native alpha bar
            naive_alpha_plot = torch.sort(naive_alpha.squeeze(), descending=True)[0].cpu().numpy()            
            plt.bar(range(self.args.train_num), naive_alpha_plot)
            plt.savefig(os.path.join(self.args.save, f"naive_alpha-{outer_epoch}.png"))
            plt.close()
            
            train_outputs = self._train(alpha=naive_alpha, state_init=ood_state_init, wandb_name=f"eval-naive-oe-{outer_epoch}")
            acc_rate_naive = self.calc_acc_rate(init_test_losses, train_outputs[-1])
            
            log_str = f"Outer Epoch: {outer_epoch} | Acc Rate: {acc_rate} | Acc Rate Naive: {acc_rate_naive}"
            print_rank(log_str)
            save_rank(log_str, os.path.join(self.args.save, "log.txt"))
            
            if dev_loss < best_dev_loss:
                best_alpha = alpha.clone()
                best_dev_loss = dev_loss
                best_outer_epoch = outer_epoch
        
        # print_rank("##### Final Evaluate #####")
        
        # print_rank(f"Best Dev Loss: {best_dev_loss}")
        # self._train(alpha=best_alpha, state_init=ood_state_init, wandb_name=f"eval-best-oe-{best_outer_epoch}")
        # torch.save(best_alpha, os.path.join(self.args.save, "best_alpha.pt"))

        # naive_best_alpha = (best_alpha > 1e-10).float()
        # naive_best_alpha = naive_best_alpha / torch.sum(naive_best_alpha)
        # self._train(alpha=naive_best_alpha, state_init=ood_state_init, wandb_name=f"eval-naive-best-oe-{best_outer_epoch}")

        # torch.save(naive_best_alpha, os.path.join(self.args.save, "naive_best_alpha.pt"))

        return best_alpha, best_outer_epoch, best_dev_loss