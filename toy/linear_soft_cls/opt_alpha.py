import torch
from tqdm import tqdm
import torch.nn as nn
import cvxpy as cp
import time
import sys
import os
import matplotlib.pyplot as plt


class OPTAlphaModel(nn.Module):
    def __init__(self,
                 dim,
                 num_steps,
                 num_alphas,
                 xn, yn,
                 dev_xn, dev_yn,
                 test_xn, test_yn,
                 eta):
        super(OPTAlphaModel, self).__init__()
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.ones(num_alphas) / num_alphas) for _ in range(num_steps)])
        self.xn = xn
        self.yn = yn.float()
        self.dev_xn = dev_xn
        self.dev_yn = dev_yn.float()
        self.test_xn = test_xn
        self.test_yn = test_yn.float()
        self.num_steps = num_steps
        self.dim = dim
        self.num_alphas = num_alphas
        self.eta = eta
        # self.register_proj_hook()

    def register_proj_hook(self):
        def proj_hook(grad):
            grad_proj = cp.Variable(grad.size(0))
            objective = cp.Minimize(cp.sum_squares(grad.squeeze().cpu().numpy() - grad_proj))
            prob = cp.Problem(objective, [cp.sum(grad_proj) == 1, grad_proj >= 0])
            result = prob.solve()
            grad_res = torch.tensor(grad_proj.value).view(grad.size()).to(grad.device)
            return grad

        for t in range(self.num_steps):
            self.alphas[t].register_hook(proj_hook)

    def inner_loss(self, x, y, theta):
        l = -y * (x @ theta) + torch.log(1 + torch.exp(x @ theta))
        l = torch.where(torch.isinf(l), (1-y) * (x @ theta), l)
        return torch.mean(l)

    def forward(self, theta, eval_mode="dev"):
        loss = 0
        losses = []
        full_losses = []
        if eval_mode == "dev":
            eval_xn, eval_yn = self.dev_xn, self.dev_yn
        else:
            eval_xn, eval_yn = self.test_xn, self.test_yn
            
        for t in tqdm(range(self.num_steps)):
            cur_loss = self.inner_loss(eval_xn, eval_yn, theta)
            full_losses.append(cur_loss.item())
            if t % 100 == 0:
                losses.append(round(cur_loss.item(), 3))
            grad_full = self.xn * (torch.sigmoid(self.xn @ theta) - self.yn)
            grad = torch.sum(self.alphas[t].unsqueeze(1) * grad_full, dim=0).unsqueeze(1)
            theta = theta - self.eta * grad
            i_loss = self.inner_loss(eval_xn, eval_yn, theta)
            loss += i_loss
        
        log_str = f"{eval_mode} Losses: {losses} Final: {i_loss}"
        loss = loss / self.num_steps

        return loss, full_losses, log_str


def proj_alpha(optimizer, args, kwargs):
    for p in optimizer.param_groups[0]["params"]:
        data = p.data
        data_proj = cp.Variable(data.size(0))
        objective = cp.Minimize(cp.sum_squares(data.squeeze().cpu().numpy() - data_proj))
        prob = cp.Problem(objective, [cp.sum(data_proj) == 1, data_proj >= 0])
        result = prob.solve()
        data_res = torch.tensor(data_proj.value).view(data.size()).to(data.device).to(data.dtype)
        p.data = data_res


def calc_acc_rate(baseline_losses, losses, steps=None):
    def _calc(step):
        if step >= len(baseline_losses):
            return -1
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


def print_and_save(log_str, base_path):
    print(log_str)
    with open(os.path.join(base_path, "log.txt"), "a") as f:
        f.write(log_str + "\n")


def solve_opt_alpha(base_path, device):
    # data_seed = "0.5-3.0-1.0-4096-10-40-1"
    data_seed = sys.argv[2]
    path = f"{base_path}/processed_data/toy_data/{data_seed}/data.pt"

    dim = 128
    num_steps = 2000
    num_alphas = 4096
    eta = 0.1
    lr = 0.002
    
    acc_steps = [int(i * num_steps / 10) for i in range(10)]
    
    base_save_path = f"{base_path}/results/toy/opt_alpha/{data_seed}-d{dim}-ns{num_steps}-na{num_alphas}-eta{eta}-lr{lr}/"
    os.makedirs(base_save_path, exist_ok=True)
    
    location = "cpu" if device == "cpu" else f"cuda:{device}"
    xn, yn, dev_xn, dev_yn, test_xn, test_yn, theta_init = torch.load(path, map_location=location)
    print(data_seed)
    print(xn)
    model = OPTAlphaModel(dim, num_steps, num_alphas, xn, yn, dev_xn, dev_yn, test_xn, test_yn, eta).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.register_step_post_hook(proj_alpha)

    area_losses = []

    epochs = 500
    with torch.no_grad():
        loss_0, bsl_losses_0, _ = model(theta_init)
        print("Area Loss 0:", loss_0.item())
        _, bsl_test_losses_0, _ = model(theta_init, eval_mode="test")
    
    area_losses.append(loss_0.item())
    
    print("#########")
    print()
    
    for epoch in range(epochs):
        st = time.time()  
        loss, losses_0, log_str = model(theta_init)
        
        print_and_save(log_str, base_save_path)
        
        with torch.no_grad():
            test_loss, test_losses_0, log_str = model(theta_init, eval_mode="test")
        
        print_and_save(log_str, base_save_path)
        
        area_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = time.time() - st
        
        log_str = "Epoch: {} | Dev Area Loss: {:.4f} | Test Area Loss: {:.4f} | Elasped Time: {}".format(
            epoch, loss.item(), test_loss.item(), elapsed)
        print_and_save(log_str, base_save_path)

        dev_acc_rate = calc_acc_rate(bsl_losses_0, losses_0, steps=acc_steps)
        test_acc_rate = calc_acc_rate(bsl_test_losses_0, test_losses_0, steps=acc_steps)
        
        log_str = f"Dev Acc Rate: {dev_acc_rate} Test Acc Rate: {test_acc_rate}"
        print_and_save(log_str, base_save_path)

        print_and_save("#" * 20 + "\n", base_save_path)
    
        sd = model.state_dict()
        alpha_t = torch.stack([sd[f"alphas.{t}"] for t in range(num_steps)], dim=0)
        # print(alpha_t)
        # print(torch.sum(alpha_t, dim=1))
        save_path = os.path.join(base_save_path, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(alpha_t, os.path.join(save_path, f"opt_alpha.pt"))
        
    torch.save(area_losses, os.path.join(base_save_path, "area_losses.pt"))
    plt.plot(area_losses)
    plt.savefig(os.path.join(base_save_path, "area_losses.png"))


def main():
    base_path = sys.argv[1]
    # device = torch.cuda.current_device()
    device = "cpu"
    solve_opt_alpha(base_path, device)
    

if __name__ == "__main__":
    main()