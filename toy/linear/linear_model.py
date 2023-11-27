import torch
from tqdm import tqdm
import os
from utils import print_rank, save_rank
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt


class LinearModel():
    def __init__(self, args, device, dim=None, path=None):
        self.args = args
        self.device = device
        self.dim = dim
        self.theta_gd = None
        self.train_data, self.test_data = None, None
        self.theta_init = None
        self.theta = None
    
    def set_theta_gd(self, path=None):
        if path is None:
            theta_gd = torch.randn(self.dim, 1, device=self.device)
        else:
            theta_gd = torch.load(path, map_location=self.device)
        self.theta_gd = theta_gd
    
    def generate_data(self, data_num):
        x = torch.randn(data_num, self.dim, device=self.device)
        x[:, 0] = 1
        y = x @ self.theta_gd
        return x, y
    
    def set_train_data(self, x, y):
        self.train_data = (x,y)
    
    def set_test_data(self, x, y):
        self.test_data = (x,y)
    
    def set_init_theta(self, theta=None):
        if theta is None:
            self.theta_init = torch.randn(self.dim, 1, device=self.device)
        else:
            self.theta_init = theta
    
    def loss(self, x, y, theta):
        loss = (x @ theta - y).pow(2).mean()
        return loss
    
    def train(self):
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data
        
        assert self.theta_init is not None
        self.theta = torch.clone(self.theta_init)
        
        all_train_loss = []
        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            loss = self.loss(train_x, train_y, self.theta)
            grad = 2 / self.args.train_num * train_x.t() @ (train_x @ self.theta - train_y) # (dim, 1)
            self.theta -= self.args.lr * grad
            
            all_train_loss.append(loss.item())
            test_loss = self.loss(test_x, test_y, self.theta)
            
            if epoch % 10 == 0:
                log_str = "Epoch: {} | Train Loss: {:.4f} | Test Loss: {:.4f}".format(epoch, loss, test_loss)
                print_rank(log_str)
            # save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                
        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        test_loss = self.loss(test_x, test_y, self.theta)
        log_str = "Final Test Loss: {}".format(test_loss)
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        torch.save(all_train_loss, os.path.join(self.args.save, "train_loss.pt"))

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
    
    def test(self):
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data
        
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
        
        
        
