import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm

from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, save_rank, get_model, all_gather
from torch.distributed import get_rank
from torch.utils.data import DataLoader, DistributedSampler

try:
    from transformers import mpu
except ImportError:
    mpu = None


class KDTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
        self.setup_teacher_model()

    def setup_teacher_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        assert args.teacher_model_path is not None
        teacher_model = get_model(args, device, args.teacher_model_path)
        teacher_model.eval()
        self.teacher_model = teacher_model

    def get_kd_loss(self, model_batch, no_model_batch, logits):
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model(**model_batch, use_cache=False)
            teacher_logits = teacher_outputs.logits
        if self.args.model_parallel:
            kd_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
            loss_mask = no_model_batch["loss_mask"]
            kd_loss = torch.sum((kd_losses * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)
        else:
            teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
            inf_mask = torch.isinf(logits)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
            x = torch.sum(prod_probs, dim=-1)
            loss_mask = no_model_batch["loss_mask"]
            kd_loss = -torch.sum(x * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
    
        return kd_loss

    def compute_lm_kd_loss(self, model_batch, no_model_batch, mean=True):
        
        if self.args.model_parallel:
            loss_func = mpu.parallel_cross_entropy
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none")
        
        outputs = self.model(**model_batch, use_cache=False)
        logits = outputs.logits
        if self.args.model_parallel:
            lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"])
        else:
            lm_losses = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            lm_losses = lm_losses.view(-1, no_model_batch["label"].size(-1))
        loss_mask = no_model_batch["loss_mask"]
        lm_loss = torch.sum((lm_losses * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)

        kd_loss = self.get_kd_loss(model_batch, no_model_batch, logits)
        
        loss = (1 - self.args.kd_ratio) * lm_loss + self.args.kd_ratio * kd_loss
        
        if mean:
            loss = torch.mean(loss)
            lm_loss = torch.mean(lm_loss)
            kd_loss = torch.mean(kd_loss)
        
        return loss, lm_loss, kd_loss

    def compute_loss(self, model_batch, no_model_batch):
        loss, lm_loss, kd_loss = self.compute_lm_kd_loss(model_batch, no_model_batch)
        dist.all_reduce(lm_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        dist.all_reduce(kd_loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        
        return loss, {"lm_loss": lm_loss.item()/self.dp_world_size, "kd_loss": kd_loss.item()/self.dp_world_size}
    
    def evaluate_kd_lm(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_lm)
        
        self.model.eval()
        all_losses, all_lm_losses, all_kd_losses = [], [], []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss, lm_loss, kd_loss = self.compute_lm_kd_loss(model_batch, no_model_batch, mean=False)
                all_losses.append(loss)
                all_lm_losses.append(lm_loss)
                all_kd_losses.append(kd_loss)
        
        all_losses = torch.cat(all_losses, dim=0)
        all_losses = all_gather(all_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_losses = all_losses.view(-1)
        avg_loss = all_losses.mean().item()

        all_lm_losses = torch.cat(all_lm_losses, dim=0)
        all_lm_losses = all_gather(all_lm_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_lm_losses = all_lm_losses.view(-1)
        avg_lm_loss = all_lm_losses.mean().item()

        all_kd_losses = torch.cat(all_kd_losses, dim=0)
        all_kd_losses = all_gather(all_kd_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_kd_losses = all_kd_losses.view(-1)
        avg_kd_loss = all_kd_losses.mean().item()

        if get_rank() == 0:
            res = {"avg_loss": avg_loss, "avg_lm_loss": avg_lm_loss, "avg_kd_loss": avg_kd_loss}
        else:
            res = None
        
        dist.barrier()
        return res

    def evaluate(self):
        if self.args.eval_ppl:
            lm_res = self.evaluate_kd_lm()
        else:
            lm_res = {}
        if self.args.eval_gen:
            prompt_ids, response_ids, gen_res, response_strs = self.evaluate_gen()
        else:
            gen_res = {}
            response_strs = []
            prompt_ids = []
            response_ids = []

        if get_rank() == 0:
            res = {**lm_res, **gen_res}

            if self.args.eval_gen:
                # for i in range(3):
                #     print_rank(f"Input:\n{self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)}\n")
                #     print_rank(f"Output:\n{response_strs[i]}\n")
                #     print_rank(f"Ground Truth:\n{self.eval_dataset.answers[i]}\n")
                #     print_rank("*" * 100)

                self.save_evals(response_ids, res, response_strs)
            
            eval_log_str = self.get_log(res, "eval")
            print_rank("*" * 100)
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
