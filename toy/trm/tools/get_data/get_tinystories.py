from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
base_path = sys.argv[1]
sys.path.append(base_path)
import json
from tqdm import tqdm
import torch
from toy.trm.tiny_story_model import ToyTokenizer

# hf_name = "roneneldan/TinyStories-1M"
# name = "tiny_stories/1M"

# save_path = os.path.join("/mnt/yuxian/checkpoints/", name)

# os.makedirs(save_path, exist_ok=True)

# tokenizer = AutoTokenizer.from_pretrained(hf_name)
# model = AutoModelForCausalLM.from_pretrained(hf_name)

# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path, safe_serialization=False)

# model = AutoModelForCausalLM.from_pretrained(save_path)

# print(' > number of parameters: {}'.format(
#     sum([p.nelement() for p in model.parameters()])), flush=True)
model_path = "/mnt/yuxian/checkpoints/mistral-7B/"
load_dir = "/mnt/yuxian/data/tinystories/all_data/"
max_length = 128

test_num = 1000
dev_num = 1000

save_dir = os.path.join(base_path, "processed_data", "toy-ts", "mistral", f"small_{max_length}")
os.makedirs(save_dir, exist_ok=True)

tokenizer = ToyTokenizer(model_path, os.path.join(load_dir, "all_tokens_mistral.pt"))

with open(os.path.join(load_dir, "data00.json")) as f:
    data = json.load(f)
    
all_data = {
    "test": data[:test_num],
    "dev": data[test_num:test_num+dev_num],
    "train": data[test_num+dev_num:]
}

for split in all_data:
    tokenized_data = []
    f_jsonl = open(os.path.join(save_dir, f"{split}.jsonl"), "w")
    for od in tqdm(all_data[split]):
        d = tokenizer.encode(od["story"])
        d = d[:max_length+1]
        if len(d) < max_length+1:
            d.extend([tokenizer.pad_token_id] * (max_length+1 - len(d)))
        tokenized_data.append(d)
        f_jsonl.write(json.dumps(od) + "\n")
    f_jsonl.close()
    tokenized_data = torch.tensor(tokenized_data)
    torch.save(tokenized_data, os.path.join(save_dir, f"{split}.pt"))
    