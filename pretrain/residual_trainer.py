import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import get_rank
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from utils import print_rank, get_model, save_rank, save_parallel, get_tokenizer, all_gather

from .trainer import PreTrainer


class ResidualModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.base_model = get_model(args, device)
        self.residual_model = get_model(args, device)
    
    def forward(self, model_batch, no_model_batch):
        base_output = self.base_model(**model_batch)
        residual_output = self.residual_model(**model_batch)
        
        return base_output.logits, residual_output.logits
    
    def save_pretrained(self, save_path):
        base_model_path = os.path.join(save_path, "base")
        os.makedirs(base_model_path, exist_ok=True)
        self.base_model.save_pretrained(base_model_path)
        residual_model_path = os.path.join(save_path, "residual")
        os.makedirs(residual_model_path, exist_ok=True)
        self.residual_model.save_pretrained(residual_model_path)


class ResidualPreTrainer(PreTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        
    def get_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        model = ResidualModel(args, device)
        return model
    
    def _compute_residual_pt_losses(self, model_batch, no_model_batch, mean=True):
        base_logits, residual_logits = self.model(model_batch, no_model_batch)
        total_logits = base_logits + residual_logits
        base_lm_loss = self._get_lm_loss_from_logits(base_logits, no_model_batch["label"], no_model_batch["loss_mask"])
        total_lm_loss = self._get_lm_loss_from_logits(total_logits, no_model_batch["label"], no_model_batch["loss_mask"])
        
        if mean:
            base_lm_loss = base_lm_loss.mean()
            total_lm_loss = total_lm_loss.mean()
        
        loss = self.args.residual_base_weight * base_lm_loss + total_lm_loss
        
        return loss, base_lm_loss, total_lm_loss
    
    def compute_loss(self, model_batch, no_model_batch):
        loss, base_lm_loss, total_lm_loss = self._compute_residual_pt_losses(model_batch, no_model_batch)
        return loss, {"base_lm_loss": base_lm_loss.item(), "total_lm_loss": total_lm_loss.item()}
    
    def evaluate(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_lm)
        
        self.model.eval()
        all_losses, all_base_losses, all_total_losses = [], [], []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss, base_lm_loss, total_lm_loss = self._compute_residual_pt_losses(model_batch, no_model_batch, mean=False)
                all_losses.append(loss)
                all_base_losses.append(base_lm_loss)
                all_total_losses.append(total_lm_loss)
        
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)
        all_base_losses = torch.cat(all_base_losses, dim=0)
        avg_base_loss = self._avg_loss_cross_dp(all_base_losses)
        all_total_losses = torch.cat(all_total_losses, dim=0)
        avg_total_loss = self._avg_loss_cross_dp(all_total_losses)

        if get_rank() == 0:
            res = {"avg_loss": avg_loss, "avg_base_loss": avg_base_loss, "avg_total_loss": avg_total_loss}
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
        else:
            res = None
        
        dist.barrier()