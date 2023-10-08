import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from kd.trainer import KDTrainer
from sft.trainer import SFTTrainer
from .modeling import MOSModel, MOSConfig


class MOSKDTrainer(KDTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
    
    def get_model(self, args, device):
        args = args or self.args
        device = device or self.device
        config = MOSConfig(args, inner_model_path=args.model_path, num_experts=(args.mos_experts or 1))
        model = MOSModel(config, device)
        if dist.get_rank() == 0:
            print('New number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)
        
        return model
    
    def compute_lm_kd_loss(self, model_batch, no_model_batch, mean=True):
                
        full_probs = self.model(**model_batch, use_cache=False)["full_probs"]
        full_logprobs = torch.log(full_probs)
        logprobs = torch.gather(full_logprobs, dim=-1, index=no_model_batch["label"].unsqueeze(-1)).squeeze(-1)
        loss_mask = no_model_batch["loss_mask"]
        lm_loss = -torch.sum((logprobs * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model(**model_batch, use_cache=False)
            teacher_logits = teacher_outputs.logits

        teacher_full_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        prod_probs = teacher_full_probs * full_logprobs
        x = torch.sum(prod_probs, dim=-1)
        kd_loss = -torch.sum(x * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        
        loss = (1 - self.args.kd_ratio) * lm_loss + self.args.kd_ratio * kd_loss
        
        if mean:
            loss = torch.mean(loss)
            lm_loss = torch.mean(lm_loss)
            kd_loss = torch.mean(kd_loss)
        
        return loss, lm_loss, kd_loss


class MOSSFTTrainer(SFTTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
    
    def get_model(self, args, device):
        args = args or self.args
        device = device or self.device
        return get_model(args, device, MOSModel, MOSConfig, config_kwargs={"num_experts": args.mos_experts})
 
    def compute_lm_loss(self, model_batch, no_model_batch, mean=True):
                
        full_probs = self.model(**model_batch, use_cache=False)["full_probs"]
        full_logprobs = torch.log(full_probs)
        logprobs = torch.gather(full_logprobs, dim=-1, index=no_model_batch["label"]).squeeze(-1)
        loss_mask = no_model_batch["loss_mask"]
        lm_loss = -torch.sum((logprobs * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)
        
        if mean:
            lm_loss = torch.mean(lm_loss)
        
        return lm_loss