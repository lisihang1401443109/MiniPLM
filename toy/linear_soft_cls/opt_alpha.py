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
        self.xn = xn
        self.yn = yn
        self.dev_xn = dev_xn
        self.dev_yn = dev_yn
        self.num_steps = num_steps
        self.dim = dim
        self.num_alphas = num_alphas
        self.eta = eta

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
