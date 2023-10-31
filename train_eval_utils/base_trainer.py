import torch
import random
import os

from utils import print_rank, get_model, save_rank, save_parallel, get_tokenizer, all_gather
from torch.distributed import get_rank
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.nn as nn
import deepspeed
import math
import numpy as np
from time import time
from collections import defaultdict
from data_utils.prompt_datasets import PromptDataset
import json
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from rouge_metric import compute_metrics

from .schedulers import WarmupCosineAnnealingLR

try:
    from transformers import mpu
except ImportError:
    mpu = None


class BaseTrainer():
    def __init__(self, args, ds_config, device, do_train=True):
        self.args = args
        self.ds_config = ds_config
        self.device = device
        self.do_train = do_train
        self.tokenizer = get_tokenizer(args)

        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None
    
    def get_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        return get_model(args, device)
    
    def get_optimizer(self, model, args=None):
        args = args or self.args
        # Use AdamW.
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps, betas=(args.adam_beta, args.adam_beta2))
        print_rank(f'Optimizer = {optimizer.__class__.__name__}')
        return optimizer
        
    def get_lr_scheduler(self, optimizer, args=None):
        args = args or self.args
        if args.lr_decay_style == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters)
        elif args.lr_decay_style == "cosine":
            lr_scheduler = WarmupCosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
                warmup_steps=args.warmup_iters,
                eta_min=args.lr_min)
        elif args.lr_decay_style == "noam":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters,
                num_training_steps=self.total_steps,
                power=0.5)
        else:
            raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

        return lr_scheduler
    
    def setup_model_and_optimizer(self, args=None, ds_config=None, device=None, set_optim=True):
        args = args or self.args
        device = device or self.device
        ds_config = ds_config or self.ds_config
        # get the model
        model = self.get_model(args, device)
        # get the optimizer and lr_scheduler
        if set_optim:
            optimizer = self.get_optimizer(model, args)
            lr_scheduler = self.get_lr_scheduler(optimizer, args)
        else:
            optimizer, lr_scheduler = None, None
            
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.model_parallel else None,
            config_params=ds_config
        )
        
        # get the memory usage
        print_rank("Model mem\n", torch.cuda.memory_summary())
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def resume_training(self):
        load_dir = self.args.resume_dir or self.args.save
        if self.args.resume_tag is None:
            with open(os.path.join(load_dir, "latest")) as f:
                tag = f.read().strip()
        else:
            tag = self.args.resume_tag
        self.model.load_checkpoint(load_dir, tag=tag)
        self.last_rng_states = torch.load(os.path.join(load_dir, tag, f"rng_states_{get_rank()}.pt"))
        
        with open(os.path.join(load_dir, tag, "dynamics.json"), "r") as f:
            dynamics = json.load(f)
        self.last_steps = dynamics["step"]
        self.last_epochs = dynamics["epoch"]
        self.last_global_steps = dynamics["global_steps"]
        self.train_dataset.set_skip_offset(dynamics["skip_offset"])
        
        print_rank(f"Resume from {load_dir} {tag}")
        print_rank(f"Resume from step {self.last_steps}, epoch {self.last_epochs}, global step {self.last_global_steps}")
 
    def prepare_learning(self, args=None):
        args = args or self.args
        self.total_batch_size = args.batch_size * self.dp_world_size * args.gradient_accumulation_steps
        self.train_iters_per_epoch = int(len(self.train_dataset) / self.total_batch_size)
        assert (args.epochs is not None) ^ (args.total_iters is not None), (args.epochs, args.total_iters)
        self.total_steps = args.total_iters or self.train_iters_per_epoch * args.epochs
        self.epochs = args.epochs or math.ceil(args.total_iters / self.train_iters_per_epoch)
        self.train_dataset.set_num(self.train_iters_per_epoch * self.total_batch_size) # droplast
        
        if args.save_interval == -1:
            args.save_interval = self.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = self.train_iters_per_epoch

        if self.args.precompute_data_order and (not self.args.resume_training):
            if get_rank() == 0:
                normal_order = np.arange(0, len(self.train_dataset), dtype=np.int32)
                order = np.stack([np.random.permutation(normal_order) for _ in range(self.epochs)], axis=0)
                order = order[:, :self.train_iters_per_epoch * self.total_batch_size] # droplast
                np.save(os.path.join(self.args.save, "data_order.npy"), order)
                print("Data order size: ", order.shape)
            dist.barrier()
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))
                        
        if self.args.resume_training:
            assert self.args.precompute_data_order
            assert os.path.exists(os.path.join(self.args.save, "data_order.npy"))
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))

        print_rank(f"Total batch size: {self.total_batch_size}")
        print_rank(f"Total iters: {self.total_steps}")
        print_rank(f"Total epochs: {self.epochs}")
        print_rank(f"Train iters per epoch: {self.train_iters_per_epoch}")
        print_rank(f"Save interval: {args.save_interval}")
        print_rank(f"Eval interval: {args.eval_interval}")
        
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = PromptDataset(args, self.tokenizer, "train", args.data_dir, args.train_num)
            print_rank("train num", len(self.train_dataset))
            self.eval_dataset = PromptDataset(args, self.tokenizer, "valid", args.data_dir, args.dev_num)
        else:
            self.eval_dataset = PromptDataset(args, self.tokenizer, "test", args.data_dir, args.dev_num)

    def compute_loss(self, model_batch, no_model_batch):
        raise NotImplementedError

    def _get_lm_loss_from_logits(self, logits, label, loss_mask):        
        if self.args.model_parallel:
            loss_func = mpu.parallel_cross_entropy
            lm_losses = loss_func(logits.contiguous().float(), label)
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_losses = loss_func(logits.float().view(-1, logits.shape[-1]), label.view(-1))
            lm_losses = lm_losses.view(-1, label.size(-1))
        lm_loss = torch.sum((lm_losses * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)
        return lm_loss

    def compute_lm_loss(self, model_batch, no_model_batch, mean=True):        
        outputs = self.model(**model_batch, use_cache=False)
        logits = outputs.logits
        
        lm_loss = self._get_lm_loss_from_logits(logits, no_model_batch["label"], no_model_batch["loss_mask"])
        
        if mean:
            lm_loss = lm_loss.mean()            
        
        return lm_loss

    def get_log(self, stats, phase, **kwargs):
        log_prefix = "{} | epoch {}/{} | steps {} | global_steps {}/{}".format(
            phase,
            self.epoch,
            self.epochs,
            self.steps,
            self.global_steps,
            self.total_steps
        )
        
        log_midfix = " | ".join([f"{k}: {v:.4f}" for k,v in stats.items()])
        log_suffix = " | ".join([(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}") for k,v in kwargs.items()])
        
        return log_prefix + " | " + log_midfix + " | " + log_suffix

    def train(self):

        train_sampler = DistributedSampler(self.train_dataset, shuffle=(not self.args.precompute_data_order), drop_last=True, rank=self.dp_rank, num_replicas=self.dp_world_size)
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate_lm, drop_last=True)

        self.steps = 0
        self.global_steps = 1
        self.epoch = 0
        
        logging_stats = defaultdict(float)
        
        if not self.args.resume_training:
            self.evaluate()
        self.model.train()

        for epoch in range(0, self.epochs):
            self.epoch = epoch
            train_sampler.set_epoch(epoch)
            self.train_dataset.set_epoch(epoch)
            print("new epoch")
            for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
                if self.args.resume_training:
                    if self.global_steps <= self.last_global_steps:
                        if (self.steps % self.args.gradient_accumulation_steps == 0) and (self.global_steps % 1000 == 0):
                            print_rank(f"Skipping global step {self.global_steps}")                        
                        self.steps += 1
                        if self.steps % self.args.gradient_accumulation_steps == 0:
                            self.global_steps += 1
                        continue
                    if (self.steps % self.args.gradient_accumulation_steps == 0) and (self.global_steps == 34484 + 1):
                        print_rank(f"Starting from global step {self.global_steps}")
                        torch.set_rng_state(self.last_rng_states["torch"])
                        torch.cuda.set_rng_state(self.last_rng_states["cuda"])
                        np.random.set_state(self.last_rng_states["numpy"])
                        random.setstate(self.last_rng_states["python"])
                
                # print_rank(f"Epoch {epochs}, Iter {it}")
                self.train_dataset.move_to_device(model_batch, no_model_batch, self.device)
                
                # if get_rank() == 0:
                #     print(model_batch["input_ids"].size(), no_model_batch["label"].size())
                
                stats = {}
                
                # forward
                forward_time = time()
                loss, loss_stats = self.compute_loss(model_batch, no_model_batch)
                stats.update(loss_stats)
                forward_time = time() - forward_time

                # backward
                backward_time = time()
                self.model.backward(loss)
                backward_time = time() - backward_time

                # step
                step_time = time()
                self.model.step()
                step_time = time() - step_time

                dist.all_reduce(loss, group=self.dp_group, op=dist.ReduceOp.SUM)
                stats["loss"] = loss / self.dp_world_size
                    
                elapsed_time = forward_time + backward_time + step_time
                stats["elasped_time"] = elapsed_time
                
                # logging
                for k in stats:
                    logging_stats[k] += stats[k]
                
                mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                
                # print first step
                if self.steps == 0:
                    print_rank(self.get_log(stats, "train",
                        lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                        scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)
                    print_rank("-" * 100)
                    print_rank("-" * 100)
                
                if (self.args.mid_log_num > 0) and ((self.steps+1) % mid_log_step == 0):
                    print_rank(self.get_log(stats, "train",
                                            lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                            scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)

                
                if (self.steps > 0) and (self.global_steps > 0) and (self.global_steps % self.args.log_interval == 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0):
                    logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                    log_str = self.get_log(logging_stats, "train", 
                                           lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                           scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0,
                                           step_time=logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps,)
                    print_rank("*" * 100)
                    print_rank(log_str)
                    print_rank(self.args.save)
                    print_rank("*" * 100)
                    save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                    logging_stats = {k:0 for k in logging_stats}

                # save
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.save_interval == 0):
                    self.save(self.args.save)

                # eval
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.eval_interval == 0):
                    self.evaluate()
                    self.model.train()

                # end
                if self.global_steps >= self.total_steps:
                    self.save(self.args.save)
                    self.evaluate()
                    return
                
                self.steps += 1
                if self.steps % self.args.gradient_accumulation_steps == 0:
                    self.global_steps += 1

    def evaluate(self):
        raise NotImplementedError

    def _avg_loss_cross_dp(self, all_losses):
        all_losses = all_gather(all_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_losses = all_losses.view(-1)
        avg_loss = all_losses.mean().item()
        return avg_loss

    def evaluate_lm(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_lm)
        
        self.model.eval()
        all_losses = []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
                all_losses.append(loss)
        
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)

        if get_rank() == 0:
            res = {"avg_loss": avg_loss}
        else:
            res = None
        
        dist.barrier()
        return res
    
    def evaluate_gen(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate_gen)
        
        self.model.eval()
        all_prompt_ids, all_response_ids = [], []
        all_gen_times = []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"Generation Evaluation", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                prompt_ids = model_batch["input_ids"]
                st = time()
                gen_out = self.generate(model_batch)
                gen_time = time() - st
                response_ids = gen_out["sequences"][:, prompt_ids.size(1):]
                all_prompt_ids.append(torch.nn.functional.pad(prompt_ids, (self.args.max_prompt_length-prompt_ids.size(1), 0), value=self.tokenizer.pad_token_id))
                all_response_ids.append(torch.nn.functional.pad(response_ids, (0, self.args.max_length-response_ids.size(1)), value=self.tokenizer.pad_token_id))
                all_gen_times.append(gen_time)
        
        all_prompt_ids = torch.cat(all_prompt_ids, dim=0)
        all_prompt_ids = all_gather(all_prompt_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_prompt_ids = all_prompt_ids.view(-1, all_prompt_ids.size(-1))
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))

        all_gen_times = all_gather(torch.tensor(all_gen_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
        gen_time = all_gen_times.sum().item()

        if get_rank() == 0:
            response_strs = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)            
            res = compute_metrics(response_strs[:len(self.eval_dataset.answers)], self.eval_dataset.answers)
        else:
            res, response_strs = None, None
        
        dist.barrier()
        return all_prompt_ids, all_response_ids, res, response_strs

    def generate(self, batch, decode_type="trm_ar"):
        max_new_tokens = self.args.max_length - batch["input_ids"].size(1)
        generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
            max_length=self.args.max_length,
            min_length=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        gen_out = self.model.generate(**batch, generation_config=generation_config)
        return gen_out
        
    def save_evals(self, preds, results, response_texts, directory = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        save_dir = os.path.join(base_ckpt_path, "eval", f"{self.global_steps}")
        os.makedirs(save_dir, exist_ok=True)
        
        if get_rank() == 0:
            torch.save(preds, os.path.join(save_dir, "preds.pt"))
            torch.save(results, os.path.join(save_dir, "results.pt"))
            with open(os.path.join(save_dir, "answers.jsonl"), "w") as f:
                for resp in response_texts:
                    f.write(json.dumps({"text": resp}) + "\n")

    def save(self, directory):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{self.global_steps}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.args.model_parallel:
            if get_rank() == 0:
                self.model.module.config.to_json_file(os.path.join(ckpt_dir, "config.json"))
                self.tokenizer.save_pretrained(ckpt_dir)
            if mpu.get_data_parallel_rank() == 0:
                save_parallel(self.model.module.base_model, ckpt_dir)
        else:
            if get_rank() == 0:
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)

            if self.args.save_all:
                self.model.save_checkpoint(base_ckpt_path, tag=f"{self.global_steps}")
                rng_states = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }
                torch.save(rng_states, os.path.join(ckpt_dir, f"rng_states_{get_rank()}.pt"))
                if get_rank() == 0:
                    with open(os.path.join(ckpt_dir, "dynamics.json"), "w") as f:
                        json.dump({
                            "step": self.steps,
                            "epoch": self.epoch,
                            "global_steps": self.global_steps,
                            "skip_offset": (self.epoch, self.global_steps * self.total_batch_size)
                        }, f)
            else:
                if get_rank() == 0:
                    self.model.module.save_pretrained(ckpt_dir)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
        
        dist.barrier()