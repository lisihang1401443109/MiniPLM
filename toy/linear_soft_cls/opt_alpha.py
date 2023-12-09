import torch
import torch.nn as nn


class OPTAlphaModel(nn.Module):
    def __init__(self,
                 dim,
                 num_steps,
                 num_alphas,
                 xn, yn,
                 dev_xn, dev_yn,
                 eta):
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.ones(num_alphas) / num_alphas) for _ in range(num_steps)])
        self.register_proj_hook()
        self.xn = xn
        self.yn = yn
        self.dev_xn = dev_xn
        self.dev_yn = dev_yn
        self.num_steps = num_steps
        self.dim = dim
        self.num_alphas = num_alphas
        self.eta = eta

    def register_proj_hook(self):
        def proj_hook(grad):
            grad_proj = cp.Variable(self.args.train_num)
            objective = cp.Minimize(cp.sum_squares(grad.squeeze().cpu().numpy() - grad_proj))
            prob = cp.Problem(objective, [cp.sum(grad_proj) == 1, grad_proj >= 0])
            result = prob.solve()
            grad_res = torch.tensor(grad_proj.value).view(grad.size()).to(grad.device)
            return grad

        for t in range(self.num_steps):
            self.alphas[t].register_hook(proj_hook)

    def inner_loss(self, theta, xn, yn):
        ln = -yn * (xn @ theta) + torch.log(1 + torch.exp(xn @ theta))
        return torch.mean(ln)

    def forward(self, theta):
        loss = 0
        for t in range(num_steps):
            grad = self.xn * (torch.sigmoid(self.xn @ theta) - self.yn)
            theta = theta - self.eta * self.alphas[t] @ grad
            loss += self.inner_loss(theta, self.dev_xn, self.dev_yn)

        loss = loss / self.num_steps

        return loss


def solve_opt_alpha(device):
    path = "data.pt"
    dim = 128
    num_steps = 10000
    num_alphas = 1024
    eta = 0.005
    xn, yn, dev_xn, dev_yn, test_xn, test_yn, theta_init = torch.load(path, map_location="cpu")
    model = OPTAlphaModel(dim, num_steps, num_alphas, xn, yn, dev_xn, dev_yn, eta)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 10
    
    for epoch in range(epochs):
        loss = model(theta_init)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        
def main():
    device = torch.cuda.current_device()
    solve_opt_alpha(device)
    

if __name__ == "__main__":
    main()