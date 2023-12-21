from trainer import ToyTrmTrainer
from logistic_trainer import LogisticTrainer
from torch.func import functional_call, grad, vmap, hessian, grad_and_value, jvp, vjp
import torch
import torch.nn as nn
import cvxpy as cp
import os
import time
from tqdm import tqdm
from utils import print_rank, save_rank
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


def proj_alpha(optimizer, args, kwargs):
    for p in tqdm(optimizer.param_groups[0]["params"], desc="Solving Projection"):
        data = p.data
        data_cpu = data.squeeze().cpu().numpy()
        data_proj = cp.Variable(data.size(0))
        objective = cp.Minimize(cp.sum_squares(data_cpu - data_proj))
        prob = cp.Problem(objective, [cp.sum(data_proj) == 1, data_proj >= 0])
        result = prob.solve()
        data_res = torch.tensor(data_proj.value).view(data.size()).to(data.device).to(data.dtype)
        p.data = data_res


class GradLayerFunction(torch.autograd.Function):   
    @staticmethod
    def clip_grad(theta, max_norm):
        if max_norm < 0:
            return theta, torch.tensor(1.0, device=theta.device)
        total_norm = torch.norm(theta)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        theta.mul_(clip_coef_clamped)
        return theta, clip_coef_clamped
     
    @staticmethod
    def forward(ctx, theta, alpha, model, xn, yn, eta, t, max_gn):
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        g, l = grad_and_value(model.compute_loss_func)(params, buffers, model, xn, yn, alpha=alpha)
        ctx.save_for_backward(theta, alpha, xn, yn)
        ctx.model = model
        ctx.eta = eta
        ctx.t = t
        
        g_vec = model.params_to_vector(g)
        g_params = model.vector_to_params(g_vec)
        
        g_vec_clipped, clip_coef = GradLayerFunction.clip_grad(g_vec, max_gn)        
        g_params_clip = model.vector_to_params(g_vec_clipped)

        new_theta = theta.clone()
        ctx.clip_coef = clip_coef
        new_theta.add_(g_vec, alpha=-eta)

        return new_theta

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.t % 100 == 0:
            print("Backward", ctx.t, ctx.eta)

        theta, alpha, xn, yn = ctx.saved_tensors
        model = ctx.model
        eta = ctx.eta
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        vmapped_grad_func = vmap(grad(model.compute_loss_func_single), in_dims=(None, None, None, 0, 0))
        vmapped_g = vmapped_grad_func(params, buffers, model, xn, yn)
                
        grad_output_params = model.vector_to_params(grad_output)
        IF_abs = torch.zeros_like(alpha)
        for n, _ in model.named_parameters():
            x1 = grad_output_params[n].view(-1)
            x2 = vmapped_g[n].contiguous().view(vmapped_g[n].size(0), -1)
            IF_abs += x2 @ x1
        
        grad_alpha = -ctx.clip_coef * IF_abs * eta

        def hvp_fwdrev(f, primals, tangents):
            def grad_wrapper(pr):
                g = grad(f)(pr, buffers, model, xn, yn, alpha=alpha)
                return g
            return jvp(grad_wrapper, primals, tangents)[1]
        
        def hvp_revrev(f, primals, tangents):
            def grad_wrapper(pr):
                g = grad(f)(pr, buffers, model, xn, yn, alpha=alpha)
                return g
            vjpfunc = vjp(grad_wrapper, primals[0])[1]
            return vjpfunc(tangents[0])[0]
        
        hvp = hvp_fwdrev(model.compute_loss_func, (params,), (grad_output_params,))
        
        hvp_vec = model.params_to_vector(hvp)
        
        theta_grad = grad_output - ctx.clip_coef * eta * hvp_vec
        
        # TODO: more accurate way to compute the gradient of alpha with clip_coef
        
        return theta_grad, grad_alpha, None, None, None, None, None, None


class DevGradLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, model, dev_xn, dev_yn):
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        dev_loss = model.compute_loss_func(params, buffers, model, dev_xn, dev_yn)
        ctx.save_for_backward(dev_xn, dev_yn)
        ctx.model = model
        ctx.params = params
        ctx.buffers = buffers
        
        return dev_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        dev_xn, dev_yn = ctx.saved_tensors
        g_dev = grad(ctx.model.compute_loss_func)(ctx.params, ctx.buffers, ctx.model, dev_xn, dev_yn)
        g_dev = ctx.model.params_to_vector(g_dev) * grad_output
        return g_dev, None, None, None, None


def constant_schedule_with_warmup(lr, n_wm_steps, t):
    if t < n_wm_steps:
        return lr * t / n_wm_steps
    else:
        return lr


class AlphaModel(nn.Module):
    def __init__(self, args, n_alpha, n_steps, n_wm_steps) -> None:
        super().__init__()
        self.args = args
        self.n_alpha = n_alpha
        self.n_steps = n_steps
        self.n_wm_steps = n_wm_steps
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.ones(n_alpha) / n_alpha) for _ in range(n_steps)])
        
    def forward(self, theta, model, xn, yn, dev_xn, dev_yn, eta, mode="dev"):
        all_losses, all_logging_losses = [], []
        area_loss = 0
        st = time.time()
        
        with torch.no_grad():
            for t in tqdm(range(self.n_wm_steps), desc=f"{mode} forward warming up"):
                cur_eta = constant_schedule_with_warmup(eta, self.args.warmup_iters, t)
                theta = GradLayerFunction.apply(
                    theta, self.alpha[t], model, xn, yn, cur_eta, t, self.args.clip_grad)
                loss = DevGradLayerFunction.apply(theta, model, dev_xn, dev_yn)
                if t % 100 == 0:
                    # print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
                    all_logging_losses.append(round(loss.item(), 4))
            
                all_losses.append(loss.item())
                area_loss += loss

        for t in tqdm(range(self.n_wm_steps, self.n_steps), desc=f"{mode} forward"):
            cur_eta = constant_schedule_with_warmup(eta, self.args.warmup_iters, t)
            theta = GradLayerFunction.apply(
                theta, self.alpha[t], model, xn, yn, cur_eta, t, self.args.clip_grad)
            loss = DevGradLayerFunction.apply(theta, model, dev_xn, dev_yn)
            if t % 100 == 0:
                # print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
                all_logging_losses.append(round(loss.item(), 4))
        
            all_losses.append(loss.item())
            area_loss += loss

        area_loss = area_loss / self.n_steps
        return area_loss, all_losses, all_logging_losses
    
    def get_trainable_params(self):
        trainable_params = []
        for n, p in self.named_parameters():
            n = n.split(".")
            if int(n[1]) >= self.n_wm_steps:
                trainable_params.append(p)
        return trainable_params

    
class OptAlphaTrainer():
    def __init__(self, args, device) -> None:
        
        self.base_trainer = ToyTrmTrainer(args, device)
        # self.base_trainer = LogisticTrainer(args, device)
        
        self.model = self.base_trainer.model
        self.train_data = self.base_trainer.train_data
        self.dev_data = self.base_trainer.dev_data
        self.test_data = self.base_trainer.test_data
        self.args = args
        self.device = device
        
        self.outer_epochs = args.outer_epochs
        self.outer_lr = args.outer_lr
        self.alpha_model = AlphaModel(args, self.train_data[0].size(0), args.epochs, args.opt_alpha_wm_steps).to(device)
        self.optimizer = torch.optim.SGD(self.alpha_model.get_trainable_params(), lr=self.outer_lr)
        self.optimizer.register_step_post_hook(proj_alpha)
        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 0)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0, self.outer_epochs)
    
    def train(self):
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        theta = self.model.params_to_vector(params)
        xn, yn = self.train_data
        dev_xn, dev_yn = self.dev_data
        test_xn, test_yn = self.test_data
        for e in range(self.outer_epochs):
            st = time.time()
            self.optimizer.zero_grad()
            area_loss, all_losses, all_logging_losses = self.alpha_model(
                theta, self.model, xn, yn, dev_xn, dev_yn, self.args.lr)
            forward_elapsed = time.time() - st
                        
            log_str = "epoch {} | dev area loss {:.4f}\n".format(e, area_loss.item())
            log_str += "All Dev Losses: {}".format(all_logging_losses)
            self.print_and_save(log_str)
            
            self.evaluate(e, theta, xn, yn, test_xn, test_yn)

            area_loss.backward()
            backward_elapsed = time.time() - st - forward_elapsed
            
            self.optimizer.step()
            self.scheduler.step()
            step_elapsed = time.time() - st - forward_elapsed - backward_elapsed
            
            log_str = "Forward Elapsed: {:.4f} | Backward Elapsed: {:.4f} | Step Elapsed: {:.4f}\n\n".format(
                forward_elapsed, backward_elapsed, step_elapsed)
            self.print_and_save(log_str)
            
            self.save(e)

    def evaluate(self, e, theta, xn, yn, test_xn, test_yn):
        with torch.no_grad():
            area_loss, all_losses, all_logging_losses = self.alpha_model(
                theta, self.model, xn, yn, test_xn, test_yn, self.args.lr, mode="test")

            log_str = "epoch {} | test area loss {:.4f}\n".format(e, area_loss.item())
            log_str += "All Test Losses: {}".format(all_logging_losses)
            self.print_and_save(log_str)
       
    def print_and_save(self, log_str):
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
         
    def save(self, epoch):
        sd = self.alpha_model.state_dict()
        alpha_t = torch.stack([sd[f"alpha.{t}"] for t in range(self.args.epochs)], dim=0)
        # print(alpha_t)
        # print(torch.sum(alpha_t, dim=1))
        save_path = os.path.join(self.args.save, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(alpha_t, os.path.join(save_path, f"opt_alpha.pt"))
              