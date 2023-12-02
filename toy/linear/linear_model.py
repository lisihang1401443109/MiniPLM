import torch
import wandb
from tqdm import tqdm
import os
import json
import time
from utils import print_rank, save_rank
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt


class LinearModel():
    def __init__(self, args, device, dim=None, path=None):
        self.args = args
        self.device = device
        self.dim = dim
        self.theta_gd = None
        self.train_data, self.dev_data = None, None
        self.theta_init = None
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.acc_rate_steps = [100, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000]
        # sum_writer_path = os.path.join(args.base_path, "runs", self.exp_name)
        # os.makedirs(sum_writer_path, exist_ok=True)
        # self.writer = SummaryWriter(log_dir=sum_writer_path)
    
    def set_theta_gd(self, path=None):
        if path is None:
            theta_gd = torch.rand(self.dim, 1, device=self.device) * self.args.linear_theta_scale
        else:
            theta_gd = torch.load(path, map_location=self.device)
        self.theta_gd = theta_gd
    
    def generate_data(self, data_num, noise_scale, x_u, x_sigma, theta_gd=None):
        x = torch.randn(data_num, self.dim, device=self.device) * x_sigma + x_u
        x[:, 0] = 1
        theta_gd = self.theta_gd if theta_gd is None else theta_gd
        y = x @ theta_gd + torch.randn(data_num, 1, device=self.device) * noise_scale
        return x, y
    
    def generate_rand_theta(self):
        return torch.randn(self.dim, 1, device=self.device)
    
    def set_train_data(self, x, y):
        self.train_data = (x,y)
    
    def set_dev_data(self, x, y):
        self.dev_data = (x,y)

    def set_test_data(self, x, y):
        self.test_data = (x, y)

    def set_init_theta(self, theta=None):
        if theta is None:
            self.theta_init = torch.randn(self.dim, 1, device=self.device)
        else:
            self.theta_init = theta
    
    def loss(self, x, y, theta, alpha=None):
        if alpha is not None:
            loss = (alpha * ((x @ theta - y)).pow(2)).sum()
        else:
            loss = (x @ theta - y).pow(2).mean()
        return loss

    def save_and_plot(self, obj, name):
        torch.save(obj, os.path.join(self.args.save, f"{name}.pt"))
        plt.plot(obj)
        plt.savefig(os.path.join(self.args.save, f"{name}.png"))
        plt.close()

    def _train(self, alpha=None, theta_init=None, IF_info=False, wandb_name="debug"):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        test_x, test_y = self.test_data
        
        run = wandb.init(
            name=f"{wandb_name}",
            project="toy-linear",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=["debug", self.args.time_stamp],)
        
        if theta_init is not None:
            theta = torch.clone(theta_init)
        else:
            assert self.theta_init is not None
            theta = torch.clone(self.theta_init)
        
        if alpha is None:
            alpha = torch.ones(self.args.train_num, 1, device=self.device)
            alpha = alpha / torch.sum(alpha)
        else:
            alpha = alpha.to(self.device)

        all_train_loss, all_dev_loss, all_test_loss = [], [], []
        for epoch in tqdm(range(self.args.epochs), desc=f"{wandb_name} Training"):
            loss = self.loss(train_x, train_y, theta, alpha)
            dev_loss = self.loss(dev_x, dev_y, theta)
            test_loss = self.loss(test_x, test_y, theta)

            grad_full = 2 * alpha * train_x * (train_x @ theta - train_y) # (train_num, dim)
            grad = torch.sum(grad_full, dim=0).unsqueeze(-1) + self.args.lam * theta # (dim, 1)
            gn = torch.norm(grad)
            
            if IF_info:
                grad_dev = 2 / self.args.dev_num * dev_x.t() @ (dev_x @ theta - dev_y)  # (dim, 1)
                IF = -grad_full @ grad_dev  # (train_num, 1)
                mean_IF = torch.mean(IF)
                var_IF = torch.var(IF)

            theta -= self.args.lr * grad
            
            wandb_log = {
                "train_loss": loss.item(),
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
            
            if epoch % self.args.log_interval == 0:
                log_str = "Epoch: {} | Train Loss: {:.4f} | Dev Loss: {:.4f} | Test Loss: {:.4f} | Grad Norm: {:.4f}".format(epoch, loss, dev_loss, test_loss, gn)
                if IF_info:
                    log_str += " | Mean IF: {:.4f} | Var IF: {:.4f}".format(mean_IF, var_IF)
                print_rank(log_str)
                # save_rank(log_str, os.path.join(self.args.save, "log.txt"))

        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        
        dev_loss = self.loss(dev_x, dev_y, theta)
        log_str = "Final Dev Loss: {}".format(dev_loss)
        print_rank(log_str)
        
        test_loss = self.loss(test_x, test_y, theta)
        log_str = "Final Test Loss: {}".format(test_loss)
        print_rank(log_str)

        run.finish()
        
        return theta, loss, dev_loss, all_train_loss, all_dev_loss, all_test_loss

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

    def get_A(self, x):
        # x: (data_num, dim)
        x = x.unsqueeze(-1) # (data_num, dim, 1)
        X = x @ x.transpose(-1, -2) # (data_num, dim, dim)
        A = 2 * torch.mean(X, dim=0)
        return A
    
    def get_b(self, x, y):
        # x: (data_num, dim)
        # y: (data_num, 1)
        b = 2 * torch.mean(x * y, dim=0)
        b = b.unsqueeze(-1) # (dim, 1)
        return b
    
    def get_int_eAtb(self, A, b):
        # integral = torch.matrix_exp(A * self.args.lr / 2) * self.args.lr
        integral = (torch.eye(self.dim, device=self.device) + torch.matrix_exp(A * self.args.lr)) / 2 * self.args.lr
        res = integral @ b
        return res
    
    def simulate(self):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        
        A = self.get_A(train_x)
        b = self.get_b(train_x, train_y)
        
        e_A = torch.matrix_exp(-A * self.args.lr)
        e_A_t = e_A
        sum_e_At = e_A
        
        int_eAtb = self.get_int_eAtb(A, b)
        
        sim_theta = self.theta_init
        
        losses = []
        
        for epoch in range(self.args.epochs):
            sim_theta = sum_e_At @ int_eAtb + e_A_t @ self.theta_init
            
            delta_theta = sim_theta - self.theta_init
            
            e_A_t = e_A_t @ e_A
            sum_e_At = sum_e_At + e_A_t
            loss = self.loss(train_x, train_y, sim_theta)
            if epoch % 10 == 0:
                print(loss)
            losses.append(loss.item())
        
        all_train_losses = torch.load(os.path.join(self.args.save, "train_loss.pt"))
        
        start = 200
        
        plt.plot(losses[start:])
        plt.plot(all_train_losses[start:])
        
        plt.savefig(os.path.join(self.args.save, "loss.png"))
        
    def test(self):
        train_x, train_y = self.train_data
        
        # Q_t, R_t = torch.linalg.qr(train_x.t(), mode="reduced")
        # train_x = Q_t.t()
        # A = self.get_A(train_x)
        
        # exp_A = torch.matrix_exp(A)
        
        # print(exp_A)
    
        train_x_1 = train_x.unsqueeze(-1)  # (data_num, dim, 1)
        train_X = 1 / train_x.size(0) * train_x_1 @ train_x_1.transpose(-1, -2)  # (data_num, dim, dim)
        
        sum_train_X = torch.sum(train_X, dim=0)
        
        print(sum_train_X)
        
        exp_A = torch.matrix_exp(sum_train_X)
        
        print(exp_A)
        
        mult_exp_A = torch.eye(self.dim, device=self.device)
        for xx in tqdm(train_X):
            mult_exp_A = mult_exp_A @ torch.matrix_exp(xx)
        
        print(mult_exp_A)
        
        print(torch.norm(exp_A - mult_exp_A))
        
        exp_A = torch.matrix_exp(0.1 * sum_train_X)
        
        T1 = sum_train_X @ exp_A
        T2 = exp_A @ sum_train_X
        
        print(torch.norm(T1 - T2))
