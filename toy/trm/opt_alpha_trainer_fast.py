from addition_trainer import ToyAdditionTrainer
from logistic_trainer import LogisticTrainer
from tiny_story_trainer import ToyTSTrainer
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
    def forward(ctx, theta, alpha, model, xn, yn, dev_xn, dev_yn, eta, t, args):
        
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}

        ctx.save_for_backward(theta, xn, yn, dev_xn, dev_yn)
        ctx.model = model
        ctx.alpha = alpha
        ctx.eta = eta
        ctx.t = t
        ctx.args = args
        
        # NOTE: compute dev loss at the beginning of each step
        dev_grad_acc_steps = dev_xn.size(0) // args.eval_batch_size
        losses = 0
        for i in range(dev_grad_acc_steps):
            dev_xn_batch = dev_xn[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
            dev_yn_batch = dev_yn[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
            loss = model.compute_loss_func(params, buffers, model, dev_xn_batch, dev_yn_batch)
            losses += loss
        dev_loss = losses / dev_grad_acc_steps
        
        if alpha is None:
            return dev_loss, None
        
        grad_acc_steps = xn.size(0) // args.batch_size
        g_vec = 0
        for i in range(grad_acc_steps):
            xn_batch = xn[i*args.batch_size:(i+1)*args.batch_size]
            yn_batch = yn[i*args.batch_size:(i+1)*args.batch_size]
            alpha_batch = alpha[i*args.batch_size:(i+1)*args.batch_size]
            g, l = grad_and_value(model.compute_loss_func)(params, buffers, model, xn_batch, yn_batch, alpha=alpha_batch)

            g_vec += model.params_to_vector(g)
        
        g_params = model.vector_to_params(g_vec)
        
        g_vec_clipped, clip_coef = GradLayerFunction.clip_grad(g_vec, args.clip_grad)        
        g_params_clip = model.vector_to_params(g_vec_clipped)

        new_theta = theta.clone()
        ctx.clip_coef = clip_coef
        new_theta.add_(g_vec, alpha=-eta)

        return dev_loss, new_theta

    @staticmethod
    def backward(ctx, loss_grad_output, grad_output):
        # print(ctx.t)
        if ctx.t % 100 == 0:
            print("Backward", ctx.t, ctx.eta)

        theta, xn, yn, dev_xn, dev_yn = ctx.saved_tensors
        alpha = ctx.alpha
        model = ctx.model
        eta = ctx.eta
        args = ctx.args
        
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        
        # 1. \partial L_{dev} / \partial \theta_{t-1}
        dev_grad_acc_steps = dev_xn.size(0) // args.eval_batch_size
        g_dev_vec = 0
        for i in range(dev_grad_acc_steps):
            dev_xn_batch = dev_xn[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
            dev_yn_batch = dev_yn[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
            g_dev = grad(ctx.model.compute_loss_func)(params, buffers, model, dev_xn_batch, dev_yn_batch)
            g_dev_vec += ctx.model.params_to_vector(g_dev)
            del g_dev
        g_dev_vec = g_dev_vec / dev_grad_acc_steps
        g_dev_vec = g_dev_vec * loss_grad_output
        
        grad_theta = g_dev_vec
        
        if alpha is None:
            # last step
            return grad_theta, None, None, None, None, None, None, None, None, None
        
        # not last step
        grad_output_params = model.vector_to_params(grad_output)
        
        # 2. \partial L / \partial \alpha_t
        vmapped_grad_func = vmap(grad(model.compute_loss_func_single), in_dims=(None, None, None, 0, 0))
        grad_acc_steps_sample = xn.size(0) // args.grad_batch_size
        IF_abs = torch.zeros_like(alpha)
        for i in range(grad_acc_steps_sample):
            xn_batch = xn[i*args.grad_batch_size:(i+1)*args.grad_batch_size]
            yn_batch = yn[i*args.grad_batch_size:(i+1)*args.grad_batch_size]
            vmapped_g = vmapped_grad_func(params, buffers, model, xn_batch, yn_batch)
            for n, _ in model.named_parameters():
                x1 = grad_output_params[n].view(-1)
                x2 = vmapped_g[n].contiguous().view(vmapped_g[n].size(0), -1)
                IF_abs[i*args.grad_batch_size:(i+1)*args.grad_batch_size] += x2 @ x1
        
        grad_alpha = -ctx.clip_coef * IF_abs * eta
        
        # 3. \partial L / \partial \theta_{t} @ \partial \theta_{t} / \partial \theta_{t-1}
        grad_acc_steps = xn.size(0) // args.batch_size
        def hvp_fwdrev(f, primals, tangents):
            def grad_wrapper(pr):
                g = {n: 0 for n in params}
                for i in range(grad_acc_steps):
                    xn_batch = xn[i*args.batch_size:(i+1)*args.batch_size]
                    yn_batch = yn[i*args.batch_size:(i+1)*args.batch_size]
                    alpha_batch = alpha[i*args.batch_size:(i+1)*args.batch_size]
                    _g = grad(f)(pr, buffers, model, xn_batch, yn_batch, alpha=alpha_batch)
                    for n in g:
                        g[n] += _g[n]
                return g
            return jvp(grad_wrapper, primals, tangents)[1]
        
        # def hvp_revrev(f, primals, tangents):
        #     def grad_wrapper(pr):
        #         g = grad(f)(pr, buffers, model, xn, yn, alpha=alpha)
        #         return g
        #     vjpfunc = vjp(grad_wrapper, primals[0])[1]
        #     return vjpfunc(tangents[0])[0]
        
        hvp = hvp_fwdrev(model.compute_loss_func, (params,), (grad_output_params,))
        
        hvp_vec = model.params_to_vector(hvp)
        
        grad_theta.add_(grad_output - ctx.clip_coef * eta * hvp_vec)

        # TODO: more accurate way to compute the gradient of alpha with clip_coef
                
        return grad_theta, grad_alpha, None, None, None, None, None, None, None, None


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
                loss, theta = GradLayerFunction.apply(
                    theta, self.alpha[t], model, xn, yn, dev_xn, dev_yn, cur_eta, t, self.args)

                if t % 100 == 0:
                    # print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
                    all_logging_losses.append(round(loss.item(), 4))
            
                all_losses.append(loss.item())
                area_loss += loss

        for t in tqdm(range(self.n_wm_steps, self.n_steps), desc=f"{mode} forward"):
            cur_eta = constant_schedule_with_warmup(eta, self.args.warmup_iters, t)
            loss, theta = GradLayerFunction.apply(
                theta, self.alpha[t], model, xn, yn, dev_xn, dev_yn, cur_eta, t, self.args)
            
            # if t % 10 == 0:
            #     print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
            if t % 100 == 0:
                # print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
                all_logging_losses.append(round(loss.item(), 4))
        
            all_losses.append(loss.item())
            area_loss += loss
        
        loss, _ = GradLayerFunction.apply(
            theta, None, model, xn, yn, dev_xn, dev_yn, eta, self.n_steps, self.args)
        area_loss += loss
        all_losses.append(loss.item())

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
        
        self.base_trainer = LogisticTrainer(args, device)
        # if args.data_names == "addition":
        #     base_trainer_cls = ToyAdditionTrainer
        # elif args.data_names == "tiny_story":
        #     base_trainer_cls = ToyTSTrainer
        # else:
        #     raise NotImplementedError
        # self.base_trainer = base_trainer_cls(args, device)
        
        self.model = self.base_trainer.model
        self.train_data = self.base_trainer.train_data
        self.dev_data = self.base_trainer.dev_data
        self.test_data = self.base_trainer.test_data
        self.args = args
        self.device = device

        if self.args.batch_size == -1:
            self.args.batch_size = self.train_data[0].size(0)
        if self.args.eval_batch_size == -1:
            self.args.eval_batch_size = self.dev_data[0].size(0)
        if self.args.grad_batch_size == -1:
            self.args.grad_batch_size = self.train_data[0].size(0)

        assert self.train_data[0].size(0) % self.args.batch_size == 0, (self.train_data[0].size(0), self.args.batch_size)
        assert self.dev_data[0].size(0) % self.args.eval_batch_size == 0, (self.dev_data[0].size(0), self.args.eval_batch_size)
        assert self.train_data[0].size(0) % self.args.grad_batch_size == 0, (self.train_data[0].size(0), self.args.grad_batch_size)
        
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
            
            # self.evaluate(e, theta, xn, yn, test_xn, test_yn)

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
              