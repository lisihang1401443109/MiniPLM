import torch
import wandb
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
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        sum_writer_path = os.path.join(args.base_path, "runs", self.exp_name)
        os.makedirs(sum_writer_path, exist_ok=True)
        # self.writer = SummaryWriter(log_dir=sum_writer_path)
    
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

    def train_iter_alpha(self):
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data
        
        alpha = torch.ones(self.args.train_num, 1, device=self.device, requires_grad=True)
        alpha = alpha / torch.sum(alpha)
        
        norm_vec = torch.ones(self.args.train_num, device=self.device)
        norm_vec = norm_vec / torch.norm(norm_vec)
        
        
        for outer_epoch in range(10):
            assert self.theta_init is not None
            self.theta = torch.clone(self.theta_init)
            
            all_train_loss = []
            wandb.init(project="toy-linear", group=self.exp_name, config=self.args)
            for epoch in tqdm(range(self.args.epochs), desc="Training"):
                loss = self.loss(train_x, train_y, self.theta, alpha)
                grad_full = 2 * alpha * train_x * (train_x @ self.theta - train_y) # (train_num, dim)
                grad = torch.sum(grad_full, dim=0).unsqueeze(-1) + self.args.lam * self.theta # (dim, 1)
                gn = torch.norm(grad)
                self.theta -= self.args.lr * grad
                
                wandb.log({"train_loss": loss.item(), "grad_norm": gn}, step=epoch)
                all_train_loss.append(loss.item())
                test_loss = self.loss(test_x, test_y, self.theta)
                
                if epoch % 100 == 0:
                    log_str = "Epoch: {} | Train Loss: {:.4f} | Test Loss: {:.4f} | Grad Norm: {:.4f}".format(epoch, loss, test_loss, gn)
                    print_rank(log_str)
                    # save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                
            log_str = "Outer Epoch {}: Final Train Loss: {}".format(outer_epoch, loss)
            print_rank(log_str)
            # save_rank(log_str, os.path.join(self.args.save, "log.txt"))
            
            test_loss = self.loss(test_x, test_y, self.theta)
            log_str = "Outer Epoch {}: Final Test Loss: {}".format(outer_epoch, test_loss)
            print_rank(log_str)
            # save_rank(log_str, os.path.join(self.args.save, "log.txt"))
            
            grad_dev = 2 / self.args.test_num * test_x.t() @ (test_x @ self.theta - test_y) # (dim, 1)
            grad_train_full = 2 * train_x * (train_x @ self.theta - train_y) # (train_num, dim)
            grad_train_full = grad_train_full + self.args.lam * self.theta.squeeze().unsqueeze(0) # (train_num, dim)
            H_full = train_x.unsqueeze(-1) @ train_x.unsqueeze(-2) + self.args.lam * torch.eye(self.dim, device=self.device).unsqueeze(0) # (train_num, dim, dim)
            inv_H_full = torch.inverse(H_full)
            grad_alpha = -(grad_train_full.unsqueeze(-2) @ inv_H_full @ grad_dev).squeeze() # (train_num, 1, 1)
            proj_grad_alpha = grad_alpha - norm_vec * (torch.dot(norm_vec, grad_alpha))
            proj_grad_alpha = proj_grad_alpha.unsqueeze(-1)
            alpha -= self.args.lr_alpha * proj_grad_alpha
            
            wandb.finish()


            
    def train_with_alpha(self):
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data
        
        # alpha = torch.rand(self.args.train_num, 1, device=self.device, requires_grad=True)
        alpha = torch.ones(self.args.train_num, 1, device=self.device, requires_grad=True)
        alpha_norm = alpha / torch.sum(alpha) * self.args.train_num
        
        assert self.theta_init is not None
        self.theta = torch.clone(self.theta_init)
        
        all_train_loss = []
        all_test_loss = []
        all_mean_IF = []
        all_var_IF = []
        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            loss = self.loss(train_x, train_y, self.theta)
            grad_full = 2 * alpha_norm * train_x * (train_x @ self.theta - train_y) # (train_num, dim)
            grad = torch.sum(grad_full, dim=0).unsqueeze(-1) # (dim, 1)
            grad_dev = 2 / self.args.test_num * test_x.t() @ (test_x @ self.theta - test_y) # (dim, 1)
            IF = grad_full @ grad_dev # (train_num, 1)
            mean_IF = torch.mean(IF)
            var_IF = torch.var(IF)
            all_mean_IF.append(mean_IF.item())
            all_var_IF.append(var_IF.item())
            
            self.theta -= self.args.lr * grad
            
            all_train_loss.append(loss.item())
            test_loss = self.loss(test_x, test_y, self.theta)
            all_test_loss.append(test_loss.item())
            
            var_IF.backward()
            print(alpha.grad)
            delta_alpha = alpha.grad - torch.ones_like(alpha.grad) * (torch.ones_like(alpha.grad).t() @ alpha.grad) / torch.norm(torch.ones_like(alpha.grad))
            print(delta_alpha)
            print(torch.sum(delta_alpha))
            exit(0)
            
            
            self.writer.add_scalar("train_loss", loss.item(), epoch)
            self.writer.add_scalar("test_loss", test_loss.item(), epoch)
            self.writer.add_scalar("mean_IF", mean_IF.item(), epoch)
            self.writer.add_scalar("var_IF", var_IF.item(), epoch)
            
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
        
        self.save_and_plot(all_train_loss, "train_loss")
        self.save_and_plot(all_test_loss, "test_loss")
        self.save_and_plot(all_mean_IF, "mean_IF")
        self.save_and_plot(all_var_IF, "var_IF")





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
