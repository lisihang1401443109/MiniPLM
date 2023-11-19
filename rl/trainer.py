import torch
import os

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import print_rank, get_model, save_rank, save_parallel, get_tokenizer, all_gather
from torch.distributed import get_rank
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import deepspeed
from torchtyping import TensorType
from time import time
from collections import defaultdict
from .storage import RolloutStorage, Batch
from .sampler import Sampler
from .rl_datasets import RLPromptDataset
import json
from tqdm import tqdm
from transformers import GenerationConfig
from rouge_metric import compute_metrics
from speculative_sampling import speculative_sampling2
from autoregressive_sampling import autoregressive_sampling


try:
    from transformers import mpu
except ImportError:
    mpu = None


class Trainer():
    def __init__(self, args, ds_config, device, do_train=True):
        self.args = args
        self.tokenizer = get_tokenizer(args)
        self.ds_config = ds_config
        self.device = device
        self.storage = RolloutStorage(args, self.tokenizer, args.seed)
        self.set_datasets(do_train=do_train)
        self.setup_model_and_optimizer(set_optim=do_train)
        self.setup_teacher_model()
        self.prepare_dataloader()
        if do_train:
            self.prepare_learning()
        
        self.sampler = Sampler(args, self, self.train_dataset)
        
    def get_optimizer(self, model, args=None):
        args = args or self.args
        # Use AdamW.
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print_rank(f'Optimizer = {optimizer.__class__.__name__}')
        return optimizer
        
    def get_lr_scheduler(self, optimizer, args=None):
        args = args or self.args
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.lr_decay_style == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters)
        elif args.lr_decay_style == "cosine":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.total_iters,
                eta_min=args.lr_min)
        elif args.lr_decay_style == "noam":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters,
                num_training_steps=args.total_iters,
                power=0.5)
        else:
            raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

        return lr_scheduler
    
    def setup_model_and_optimizer(self, args=None, ds_config=None, device=None, set_optim=True):
        args = args or self.args
        device = device or self.device
        ds_config = ds_config or self.ds_config
        # get the model
        model = get_model(args, device)
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
    
    def setup_teacher_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        teacher_model = get_model(args, device, args.teacher_model_path)
        teacher_model.eval()
        self.teacher_model = teacher_model
    
    def prepare_dataloader(self, args=None):
        args = args or self.args
        self.train_dataloader = self.storage.create_dataloader(
            batch_size=self.args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
        )
        
        self.eval_dataloader = self.eval_dataset.create_dataloader(
            batch_size=self.args.eval_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    def prepare_learning(self, args=None):
        args = args or self.args
        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None
        
        args = args or self.args
        tot_iters_per_epoch = int(args.num_rollouts / (args.gradient_accumulation_steps * args.batch_size))
        self.total_steps = int(
            args.epochs
            * args.inner_epochs
            * tot_iters_per_epoch
        )
        self.total_steps = min(self.total_steps, args.total_iters)
        
        # print information
        print_rank(f"Total steps: {self.total_steps}")
        print_rank(f"Total epochs: {args.epochs}")
        print_rank(f"Total inner epochs: {args.inner_epochs}")
        print_rank(f"Total iters per epoch: {tot_iters_per_epoch}")
        print_rank(f"Total iters per inner epoch: {tot_iters_per_epoch * args.inner_epochs}")
    
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = RLPromptDataset(args, self.tokenizer, "train", args.data_dir, args.train_num)
            print_rank("train num", len(self.train_dataset))
            self.eval_dataset = RLPromptDataset(args, self.tokenizer, "valid", args.data_dir, args.dev_num)
        else:
            self.eval_dataset = RLPromptDataset(args, self.tokenizer, "valid", args.data_dir, args.dev_num)

    def get_model_inputs(self, batch: Batch):
        full_ids = torch.cat([batch.prompt_ids, batch.response_ids], dim=-1)
        input_ids = full_ids[:, :-1]
        labels = full_ids[:, 1:]
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        loss_mask = (input_ids != self.tokenizer.pad_token_id).float()
        loss_mask[:, :batch.prompt_ids.size(1)-1] = 0
        
        model_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        no_model_batch = {
            "labels": labels,
            "loss_mask": loss_mask,
        }
        
        if (self.args.model_type in ["gpt2"]):
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
            model_batch["position_ids"] = position_ids
                
        return model_batch, no_model_batch

    def compute_loss(self, model_batch, no_model_batch, mean=True):
        model_outputs = self.model(**model_batch, return_dict=True, use_cache=False)
        logits = model_outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**model_batch, return_dict=True, use_cache=False)
            teacher_logits = teacher_outputs.logits
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            
        losses: TensorType["batch_size", "full_length"] = 1 - torch.sum(torch.min(probs, teacher_probs), dim=-1)        
        loss: TensorType["batch_size"] = torch.sum((losses * no_model_batch["loss_mask"]), -1) / torch.sum(no_model_batch["loss_mask"], dim=-1)
        
        if mean:
            loss = torch.mean(loss, dim=-1)
        
        return loss

    def get_log(self, stats, phase, **kwargs):
        log_prefix = "{} | epoch {} | inner_epoch {} | steps {} | global_steps {}".format(
            phase,
            self.epoch,
            self.inner_epoch,
            self.steps,
            self.global_steps,
        )
        
        log_midfix = " | ".join([f"{k}: {v:.4f}" for k,v in stats.items()])
        log_suffix = " | ".join([f"{k}: {v:.4f}" for k,v in kwargs.items()])
        
        return log_prefix + " | " + log_midfix + " | " + log_suffix

    def train(self):
        
        self.steps = 1
        self.global_steps = 1
        self.epoch = 1
        self.inner_epoch = 1
        
        logging_stats = defaultdict(float)
        
        self.evaluate()
        self.storage.clear_history()
        self.sampler.run_sample(self.args.num_rollouts, self.global_steps)
        for epoch in range(self.args.epochs):
            for inner_epoch in range(self.args.inner_epochs):
                self.epoch = epoch
                self.inner_epoch = inner_epoch
                for it, batch in enumerate(self.train_dataloader):
                    # print_rank(f"Epoch {epochs}, Iter {it}")
                    self.storage.move_to_device(batch, self.device)

                    stats = {}
                    
                    if self.args.model_parallel:
                        self.storage.broadcast(batch, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
                        
                    model_batch, no_model_batch = self.get_model_inputs(batch)
                    forward_time = time()
                    loss = self.compute_loss(model_batch, no_model_batch)
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
                    # save
                    if self.steps % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_steps < 10000 and (self.global_steps % 1000 == 0)) or \
                        self.global_steps % self.args.save_interval == 0):
                        self.save(self.args.save)

                    # eval
                    if self.steps % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_steps < 1000 and (self.global_steps % 100 == 0)) or \
                        (self.global_steps % self.args.eval_interval == 0)):
                        self.evaluate()
                        self.model.train()
                        
                    elapsed_time = forward_time + backward_time + step_time
                    stats["elasped_time"] = elapsed_time
                    
                    for k in stats:
                        logging_stats[k] += stats[k]
                        
                    mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                    mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                    if self.steps % mid_log_step == 0:
                        print_rank(self.get_log(stats, "train"))
                    
                    if self.global_steps % self.args.log_interval == 0 and self.steps % self.args.gradient_accumulation_steps == 0:
                        logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                        log_str = self.get_log(logging_stats, "train", step_time=logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps)
                        print_rank("*" * 100)
                        print_rank(log_str)
                        print_rank(self.args.save)
                        print_rank("*" * 100)
                        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                        logging_stats = {k:0 for k in logging_stats}

                    # end
                    if self.global_steps >= self.total_steps:
                        self.save(self.args.save)
                        self.evaluate()
                        return
                    
                    self.steps += 1
                    if self.steps % self.args.gradient_accumulation_steps == 0:
                        self.global_steps += 1
                        
                self.post_backward_callback()

            self.post_epoch_callback(self.epoch)

    def post_backward_callback(self):
        pass
        
    def post_epoch_callback(self, epoch):
        self.storage.clear_history()
        # self.store.load(self.args.save)
        self.sampler.run_sample(
            self.args.num_rollouts, self.global_steps
        )  # Collect more rollouts for training

    def evaluate(self, decode_type=None):
        decode_type = decode_type or self.args.decode_type
        self.model.eval()
        all_prompt_ids, all_response_ids = [], []
        all_tvds = []
        all_gen_times = []
        
        if decode_type == "sp":
            all_acc_times, all_rej_times = [], []
            eval_dataloader = self.eval_dataset.create_dataloader(batch_size=1, shuffle=False, num_workers=self.args.num_workers, drop_last=False)
            print_rank("WARNING: speculative sampling needs eval_batch_size = 1")      
        else:
            eval_dataloader = self.eval_dataloader
            
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, f"Generation Evaluation {decode_type}", disable=(not get_rank() == 0)):
                self.eval_dataset.move_to_device(batch, self.device)
                prompt_ids = batch.input_ids
                st = time()
                gen_out = self.generate(batch.__dict__, decode_type=decode_type)
                gen_time = time() - st
                response_ids = gen_out["sequences"][:, prompt_ids.size(1):]
                model_inputs = self.get_model_inputs(Batch(prompt_ids, response_ids))
                tvd = self.compute_loss(*model_inputs, mean=False)
                all_prompt_ids.append(torch.nn.functional.pad(prompt_ids, (self.args.max_prompt_length-prompt_ids.size(1), 0), value=self.tokenizer.pad_token_id))
                all_response_ids.append(torch.nn.functional.pad(response_ids, (0, self.args.max_length-response_ids.size(1)), value=self.tokenizer.pad_token_id))
                all_tvds.append(tvd)
                all_gen_times.append(gen_time)
                if decode_type == "sp":
                    all_acc_times.append(gen_out["acc_times"])
                    all_rej_times.append(gen_out["rej_times"])
        
        all_prompt_ids = torch.cat(all_prompt_ids, dim=0)
        all_prompt_ids = all_gather(all_prompt_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_prompt_ids = all_prompt_ids.view(-1, all_prompt_ids.size(-1))
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))

        all_tvds = torch.cat(all_tvds, dim=0)
        all_tvds = all_gather(all_tvds, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_tvds = all_tvds.view(-1)
        tvd = torch.mean(all_tvds, dim=0).item()

        all_gen_times = all_gather(torch.tensor(all_gen_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
        gen_time = all_gen_times.sum().item()

        if decode_type == "sp":
            all_acc_times = all_gather(torch.tensor(all_acc_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
            all_rej_times = all_gather(torch.tensor(all_rej_times, device=self.device), dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack").view(-1)
            acc_times = all_acc_times.sum().item()
            rej_times = all_rej_times.sum().item()

        if get_rank() == 0:
            response_strs = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
            tot_gen_tokens = sum([len(self.tokenizer.encode(s, add_special_tokens=False)) for s in response_strs])
            tokens_per_sec = tot_gen_tokens / gen_time
            for i in range(3):
                print_rank(f"Input:\n{self.tokenizer.decode(all_prompt_ids[i], skip_special_tokens=True)}\n")
                print_rank(f"Output:\n{response_strs[i]}\n")
                print_rank(f"Ground Truth:\n{self.eval_dataset.answers[i]}\n")
                print_rank(f"TVD: {all_tvds[i]}, Time: {all_gen_times[i]}")
                if decode_type == "sp":
                    print_rank(f"Acc times: {all_acc_times[i]}, Rej times: {all_rej_times[i]}")
                print_rank("*" * 100)
            
            res = compute_metrics(response_strs[:len(self.eval_dataset.answers)], self.eval_dataset.answers)
            self.save_evals(all_response_ids, res, response_strs)
            stats = {"tvd": tvd, "tokens/sec": tokens_per_sec, **res}
            if decode_type == "sp":
                stats["acc_times"] = acc_times
                stats["rej_times"] = rej_times
                stats["acc_rate"] = acc_times / (acc_times + rej_times)
                stats["rej_rate"] = rej_times / (acc_times + rej_times)

            eval_log_str = self.get_log(stats, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
        
        dist.barrier()

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
        if decode_type == "trm_ar":
            gen_out = self.model.generate(**batch, generation_config=generation_config)
        elif decode_type == "sp":
            gen_out = speculative_sampling2(self.teacher_model, self.model, **batch, generation_config=generation_config, lookahead=self.args.lookahead)
        elif decode_type == "ar":
            gen_out = autoregressive_sampling(self.model, **batch, generation_config=generation_config)
        else:
            raise NotImplementedError
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
                self.model.module.base_model.save_pretrained(ckpt_dir, safe_serialization=False)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)