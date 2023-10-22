import torch
import os

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import print_rank, get_model, save_rank, save_parallel, get_tokenizer, all_gather
from torch.distributed import get_rank
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import deepspeed
import math
from time import time
from collections import defaultdict
from data_utils.prompt_datasets import PromptDataset
import json
from tqdm import tqdm
from transformers import GenerationConfig
from rouge_metric import compute_metrics

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
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print_rank(f'Optimizer = {optimizer.__class__.__name__}')
        return optimizer
        
    def get_lr_scheduler(self, optimizer, args=None):
        args = args or self.args
        if args.lr_decay_style == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters)
        elif args.lr_decay_style == "cosine":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
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
            
    def prepare_learning(self, args=None):
        args = args or self.args
        self.train_iters_per_epoch = int(len(self.train_dataset) / (args.batch_size * self.dp_world_size * args.gradient_accumulation_steps))
        assert (args.epochs is not None) ^ (args.total_iters is not None), (args.epochs, args.total_iters)
        self.total_steps = args.total_iters or self.train_iters_per_epoch * args.epochs
        self.epochs = args.epochs or math.ceil(args.total_iters / args.train_iters_per_epoch)
        
        if args.save_interval == -1:
            args.save_interval = self.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = self.train_iters_per_epoch
            
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

    def compute_lm_loss(self, model_batch, no_model_batch, mean=True):
        
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

        train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True, rank=self.dp_rank, num_replicas=self.dp_world_size)
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate_lm)

        self.steps = 1
        self.global_steps = 1
        self.epoch = 1
        
        logging_stats = defaultdict(float)
        
        self.evaluate()
        self.model.train()
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            train_sampler.set_epoch(epoch)
            for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
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
                if self.steps % mid_log_step == 0:
                    print_rank(self.get_log(stats, "train",
                                            lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                            scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)
                
                if self.global_steps % self.args.log_interval == 0 and self.steps % self.args.gradient_accumulation_steps == 0:
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
                if (self.steps % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.save_interval == 0):
                    self.save(self.args.save)

                # eval
                if (self.steps % self.args.gradient_accumulation_steps == 0) and \
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
        all_losses = all_gather(all_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_losses = all_losses.view(-1)
        avg_loss = all_losses.mean().item()

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
                self.model.module.save_pretrained(ckpt_dir)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)