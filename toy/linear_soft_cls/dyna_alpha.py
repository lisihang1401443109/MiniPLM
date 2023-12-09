import torch
import wandb
from tqdm import tqdm
import os
import json
import time
from scipy.stats import pearsonr, spearmanr
from utils import print_rank, save_rank
from matplotlib import pyplot as plt
from linear_cls_model import LinearCLSModel
import cvxpy as cp


class LinearCLSModelDynaAlpha(LinearCLSModel):
    def __init__(self, args, device, dim=None, real_dim=None, path=None):
        super(LinearCLSModelDynaAlpha, self).__init__(args, device, dim, path)

    def get_correlation(self, x, y):
        return round(pearsonr(x.cpu().numpy(), y.cpu().numpy())[0], 3)

    def train(self):
        train_x, train_y = self.train_data
        dev_x, dev_y = self.dev_data
        test_x, test_y = self.test_data
                
        ood_test_x, ood_test_y = self.generate_data(
            self.args.dev_num, self.args.dev_noise, -0.1, self.args.dev_sigma)
        
        print("Baseline")
        
        baseline_out = self._train(wandb_name="baseline", IF_info=True)
        baseline_dev_losses = baseline_out[-2]
        baseline_test_losses = baseline_out[-1]
        baseline_theta = baseline_out[0]
        
        ood_baseline_test_loss = self.loss(ood_test_x, ood_test_y.float(), baseline_theta)
        print_rank("OOD Baseline Test Loss: {}".format(ood_baseline_test_loss.item()))
        avg_train_loss = self.loss(train_x, train_y.float(), baseline_theta)
        print_rank("Avg Train Loss: {}".format(avg_train_loss.item()))

        run = wandb.init(
            name=f"dyna_alpha",
            project="toy-linear",
            group=self.exp_name,
            config=self.args,
            reinit=True,
            tags=[self.args.time_stamp],)

        assert self.theta_init is not None
        theta = torch.clone(self.theta_init)

        alpha = torch.ones(self.args.train_num, 1, device=self.device)
        origin_alpha = torch.clone(alpha)
        alpha = alpha / torch.sum(alpha)

        norm_vec = torch.ones(self.args.train_num, device=self.device)
        norm_vec = norm_vec / torch.norm(norm_vec)

        all_train_loss, all_dev_loss, all_test_loss = [], [], []
        best_dev_loss = float("inf")
        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            loss = self.loss(train_x, train_y.float(), theta, alpha)
            loss_no_alpha = self.loss(train_x, train_y.float(), theta)
            dev_loss = self.loss(dev_x, dev_y.float(), theta)
            test_loss = self.loss(test_x, test_y.float(), theta)
            
            grad_full_no_alpha = train_x * (self.soft_f(train_x @ theta) - train_y.float()) # (train_num, dim)
            
            grad_dev = 1 / self.args.dev_num * dev_x.t() @ (self.soft_f(dev_x @ theta) - dev_y.float()) # (dim, 1)
            IF = -grad_full_no_alpha @ grad_dev # (train_num, 1)
            mean_IF = torch.mean(IF)
            weighted_mean_IF = torch.sum(alpha * IF)
            var_IF = torch.var(IF)
            ratio = torch.abs(mean_IF) / (torch.sqrt(var_IF) + 1e-8)
            weighted_ratio = torch.abs(weighted_mean_IF) / (torch.sqrt(var_IF) + 1e-8)

            if epoch % self.args.alpha_update_interval == 0:
                
                if self.args.approx_proj:                
                    delta_alpha = IF.squeeze() - norm_vec * (torch.dot(norm_vec, IF.squeeze()))
                    delta_alpha = delta_alpha.unsqueeze(-1)

                    delta_alpha_norm = torch.norm(delta_alpha)
                    
                    alpha -= self.args.lr_alpha * delta_alpha

                    alpha = torch.clamp(alpha, min=0)
                    alpha = alpha / torch.sum(alpha)
                else:                
                    alpha_before_proj = alpha - self.args.lr_alpha * IF
                    alpha_proj = cp.Variable(self.args.train_num)
                    objective = cp.Minimize(cp.sum_squares(alpha_before_proj.squeeze().cpu().numpy() - alpha_proj))
                    prob = cp.Problem(objective, [cp.sum(alpha_proj) == 1, alpha_proj >= 0])
                    result = prob.solve()
                    alpha = torch.tensor(alpha_proj.value).unsqueeze(-1).to(self.device)
                
                # if epoch <= 100 or (epoch <= 1000 and epoch % 100 == 0) or (epoch % 1000 == 0):

                #     # plt.plot(range(len(raw_grad_norm)), raw_grad_norm.cpu().numpy(), label="raw_grad_norm")
                #     # plt.savefig(os.path.join(self.args.save, f"raw_grad_norm-{epoch}.png"))
                #     # plt.close()
                #     torch.save(alpha, os.path.join(self.args.save, f"alpha-{epoch}.pt"))
                #     # sorted_alpha = torch.sort(alpha.squeeze(), descending=True)[0]
                #     # print(sorted_noise_index)
                #     # sorted_alpha = alpha[sorted_noise_index]
                #     plt.plot(range(len(alpha)), alpha.squeeze().cpu().numpy(), label="alpha")
                #     # plt.plot(range(len(sorted_alpha)), sorted_alpha.squeeze().cpu().numpy(), label="sorted_alpha")
                #     plt.legend()
                #     plt.savefig(os.path.join(self.args.save, f"alpha-{epoch}.png"))
                #     plt.close()
                                        
            # if epoch < 10:
            #     torch.save(alpha, os.path.join(self.args.save, f"alpha-{epoch}.pt"))
            #     sorted_alpha = torch.sort(alpha.squeeze(), descending=True)[0]
            #     plt.bar(range(len(sorted_alpha)), sorted_alpha.cpu().numpy())
            #     plt.savefig(os.path.join(self.args.save, f"alpha-{epoch}.png"))
            #     plt.close()
            
            grad_full = alpha * grad_full_no_alpha # (train_num, dim)
            grad = torch.sum(grad_full, dim=0).unsqueeze(-1) + self.args.lam * theta # (dim, 1)
            theta -= self.args.lr * grad
            
            grad_norm = torch.norm(grad)

            train_acc = self.acc(train_x, train_y, theta)
            dev_acc = self.acc(dev_x, dev_y, theta)
            test_acc = self.acc(test_x, test_y, theta)

            # train_grad_norm = torch.norm(grad_full_no_alpha, dim=1)
            # dev_grad_norm = torch.norm(grad_dev)
            # cos_train_dev = -IF / (train_grad_norm * dev_grad_norm + 1e-8).unsqueeze(-1)
            
            # corr_train_grad = self.get_correlation(train_grad_norm, alpha.squeeze())
            # corr_cos_train_dev = self.get_correlation(cos_train_dev.squeeze(), alpha.squeeze())

            wandb.log({
                "train_loss": loss.item(),
                "train_loss_no_alpha": loss_no_alpha.item(),
                "dev_loss": dev_loss.item(),
                "test_loss": test_loss.item(),
                "train_acc": train_acc.item(),
                "dev_acc": dev_acc.item(),
                "test_acc": test_acc.item(),
                "mean_IF": mean_IF.item(),
                "var_IF": var_IF.item(),
                "std_IF": torch.sqrt(var_IF).item(),
                "ratio": ratio.item(),
                "weighted_mean_IF": weighted_mean_IF.item(),
                "weighted_ratio": weighted_ratio.item(),
                # "delta_alpha_norm": delta_alpha_norm.item(),
                # "corr_train_grad": corr_train_grad,
                # "corr_cos_train_dev": corr_cos_train_dev,
            })

            all_train_loss.append(loss.item())
            all_dev_loss.append(dev_loss.item())
            all_test_loss.append(test_loss.item())

            if epoch % self.args.log_interval == 0:
                log_str = "Epoch: {} | Train Loss: {:.4f} | Train Loss no Alpha: {:.4f} | Dev Loss: {:.4f} | Test Loss: {:.4f}".format(
                    epoch, loss, loss_no_alpha, dev_loss, test_loss)
                log_str += " | Train Acc: {:.4f} | Dev Acc: {:.4f} | Test Acc: {:.4f}".format(
                    train_acc, dev_acc, test_acc)
                log_str += " | Mean IF: {:.4f} | Var IF: {:.4f}".format(mean_IF, var_IF)
                # log_str += " | Delta Alpha Norm: {:.4f}".format(delta_alpha_norm)
                log_str += " | Grad Norm: {:.4f}".format(grad_norm)
                print_rank(log_str)
                
            # if dev_loss < best_dev_loss:
            #     best_dev_loss = dev_loss
            # else:
            #     print_rank("Early stop at epoch {}".format(epoch))
            #     break
            # if dev_loss < 0.002:
            #     print_rank("Early stop at epoch {}".format(epoch))
            #     break
        
        all_train_loss = all_train_loss + [all_train_loss[-1]] * (self.args.epochs - len(all_train_loss))
        all_dev_loss = all_dev_loss + [all_dev_loss[-1]] * (self.args.epochs - len(all_dev_loss))
        all_test_loss = all_test_loss + [all_test_loss[-1]] * (self.args.epochs - len(all_test_loss))
        
        log_str = "Final Train Loss: {}".format(loss)
        print_rank(log_str)
        
        dev_loss = self.loss(dev_x, dev_y.float(), theta)
        log_str = "Final Dev Loss: {}".format(dev_loss)
        print_rank(log_str)
        
        test_loss = self.loss(test_x, test_y.float(), theta)
        log_str = "Final Test Loss: {}".format(test_loss)
        print_rank(log_str)
        
        ood_test_loss = self.loss(ood_test_x, ood_test_y.float(), theta)
        print_rank("OOD Test Loss: {}".format(ood_test_loss.item()))
        
        avg_train_loss = self.loss(train_x, train_y.float(), theta)
        print_rank("Avg Train Loss: {}".format(avg_train_loss.item()))
        
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
