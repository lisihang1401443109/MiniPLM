from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
try:
    from transformers import mpu
except:
    print("WARNING: mpu not found")

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, get_model
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling, speculative_sampling2

from rouge_metric import compute_metrics
import deepspeed
import time

torch.set_num_threads(4)


def prepare_dataset_sp(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def compute_loss(args, model, input_ids, attention_mask, label_ids):
    if args.model_type in ["gpt2"]:
        position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
        out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = out.logits
    loss_mask = (label_ids != -100).float()
    if args.model_parallel:
        lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
        lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
    else:
        loss_func = nn.CrossEntropyLoss(reduction="none")
        lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
        lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
    
    return lm_loss


def compute_neg_tvd(args, tokenizer, draft_model, target_model, prompt_ids, response_ids):
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    input_ids = full_ids [:, :-1]
    model_batch = {"input_ids": input_ids}
    model_batch["attention_mask"] = (input_ids != tokenizer.pad_token_id)
    if (args.model_type in ["gpt2"]):
        position_ids = torch.cumsum(model_batch["attention_mask"], dim=-1) - 1
        position_ids.masked_fill_(~model_batch["attention_mask"], 0)
        model_batch["position_ids"] = position_ids

    loss_mask = (input_ids != tokenizer.pad_token_id).float()
    loss_mask[:, :prompt_ids.size(1)-1] = 0

    draft_model_outputs = draft_model(**model_batch, return_dict=True, use_cache=False)
    draft_logits = draft_model_outputs.logits
    draft_probs = torch.softmax(draft_logits, dim=-1)
    
    target_outputs = target_model(**model_batch, return_dict=True, use_cache=False)
    target_logits = target_outputs.logits
    target_probs = torch.softmax(target_logits, dim=-1)
        
    tvds = 1 - torch.sum(torch.min(draft_probs, target_probs), dim=-1)        
    tvd = torch.sum((tvds * loss_mask), -1) / torch.sum(loss_mask, dim=-1)
    neg_tvd = 1 - tvd 
    return neg_tvd

def run_model(args, tokenizer, model, draft_model, dataset: PromptDataset, epoch, device):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses, all_draft_lm_losses = [], []
    all_tvds = []
    tot_gen_time = 0
    
    generation_config = GenerationConfig(
        temperature=args.temperature,
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    acc_times, rej_times = 0, 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  

            if args.eval_ppl:
                lm_loss = compute_loss(args, model, input_ids, attention_mask, label_ids)
                all_lm_losses.append(lm_loss)
                if draft_model is not None:
                    draft_lm_loss = compute_loss(args, draft_model, input_ids, attention_mask, label_ids)
                    all_draft_lm_losses.append(draft_lm_loss)

            if args.eval_gen:
                query_ids = model_batch["input_ids"]
                st = time.time()
                if args.decode_type == "ar":
                    gen_out = autoregressive_sampling(
                        model,
                        **model_batch,
                        generation_config=generation_config)
                elif args.decode_type == "sp":
                    # gen_out = speculative_sampling(it, model, draft_model, query_ids, args.max_length, tokenizer, temperature=args.temperature, debug=False)
                    gen_out = speculative_sampling2(model, draft_model, **model_batch, generation_config=generation_config, lookahead=args.lookahead)
                    acc_times += gen_out["acc_times"]
                    rej_times += gen_out["rej_times"]
                else:
                    raise NotImplementedError
                ed = time.time()
                full_ids = gen_out["sequences"]
                response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
                if args.eval_tvd:
                    tvd = compute_neg_tvd(args, tokenizer, draft_model, model, query_ids, response_ids)
                    all_tvds.append(tvd)
                all_query_ids.extend(query_ids)
                all_response_ids.extend(response_ids)
                
                tot_gen_time += ed - st
                # print(tokenizer.batch_decode(response_ids, skip_special_tokens=True))

    additional_stats = {"acc_times": acc_times, "rej_times": rej_times, "acc_rate": acc_times / (acc_times + rej_times + 1e-5)}
    
    if args.eval_ppl:
        all_lm_losses = torch.cat(all_lm_losses, dim=0)
        mean_lm_loss = all_lm_losses.mean().item()
        if draft_model is not None:
            all_draft_lm_losses = torch.cat(all_draft_lm_losses, dim=0)
            mean_draft_lm_loss = all_draft_lm_losses.mean().item()
        else:
            mean_draft_lm_loss = 0
    else:
        mean_lm_loss = 0
        mean_draft_lm_loss = 0

    if args.eval_tvd:
        all_tvds = torch.cat(all_tvds, dim=0)
        mean_tvd = all_tvds.mean().item()
    else:
        mean_tvd = 0

    return (
        mean_lm_loss,
        mean_draft_lm_loss,
        mean_tvd,
        all_query_ids,
        all_response_ids,
        tot_gen_time,
        additional_stats)


def setup_model(args, ds_config, device, model_path=None):
    # get the model
    model = get_model(args, device, model_path=model_path)
    # get the optimizer and lr_scheduler

    optimizer, lr_scheduler = None, None
        
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model


def evaluate_sp(args, tokenizer, model, dataset: PromptDataset, split, epoch, device, ds_config):
    
    if args.decode_type == "sp":
        draft_model = setup_model(args, ds_config, device, model_path=args.draft_model_path)
    else:
        draft_model = None
    
    lm_loss, draft_lm_loss, tvd, query_ids, response_ids, tot_gen_time, additional_stats = run_model(args, tokenizer, model, draft_model, dataset, epoch, device)
    
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    tot_gen_tokens = sum([len(tokenizer.encode(s, add_special_tokens=False)) for s in response_strs])
    tokens_per_sec = tot_gen_tokens / tot_gen_time

    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:    
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {gen_res} | lm_loss {round(lm_loss, 4)} | draft_lm_loss {round(draft_lm_loss, 4)} | tvd {round(tvd, 4)} | avg. gen lenth: {mean_gen_length} | tokens/sec: {round(tokens_per_sec, 2)} | " + \
              f"{additional_stats}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
