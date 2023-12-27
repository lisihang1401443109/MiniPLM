from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import json
from tqdm import tqdm
import torch
from tiny_story_model import ToyTokenizer
from collections import Counter
import random


base_path = sys.argv[1]
model_path = "/mnt/yuxian/checkpoints/mistral-7B/"
load_dir = "/mnt/yuxian/data/tinystories/all_data/"
max_length = 64

test_num = 500
dev_num = 500
train_num = 4000
max_vocab_size = 4000

seed = 42

save_dir = os.path.join(base_path, "processed_data", "toy-ts", "mistral", f"small_{max_length}_{train_num}_{dev_num}_2")
os.makedirs(save_dir, exist_ok=True)

tokenizer_orig = AutoTokenizer.from_pretrained(model_path)
tokenizer_orig.pad_token = tokenizer_orig.eos_token

with open(os.path.join(load_dir, "data00.json")) as f:
    data = json.load(f)

data = [d for d in tqdm(data, "preprocessing") if len(d["story"].strip()) > 10]

random.seed(seed)
random.shuffle(data)

all_data = {
    "test": data[:test_num],
    "dev": data[test_num:test_num+dev_num],
    "train": data[test_num+dev_num:test_num+dev_num+train_num]
}

all_data_tokens = {
    "test": set(),
    "dev": set(),
    "train": set()
}

all_tokenized_data = {}

all_tokens = Counter()

for split in all_data:
    tokenized_data = []
    f_jsonl = open(os.path.join(save_dir, f"{split}.jsonl"), "w")
    for od in tqdm(all_data[split]):
        d = tokenizer_orig.encode(od["story"].strip(), add_special_tokens=False)
        d = d[:max_length+1]
        
        all_tokens.update(d)
        all_data_tokens[split].update(d)
        
        if len(d) < 2:
            print(od)
        
        if len(d) < max_length+1:
            d.extend([tokenizer_orig.pad_token_id] * (max_length+1 - len(d)))
        tokenized_data.append(d)
        f_jsonl.write(json.dumps(od) + "\n")
    f_jsonl.close()
    all_tokenized_data[split] = tokenized_data
    tokenized_data = torch.tensor(tokenized_data)
    torch.save(tokenized_data, os.path.join(save_dir, f"{split}_orig.pt"))

all_tokens_sorted = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)

print([(x, tokenizer_orig.convert_ids_to_tokens([x])[0], c) for x,c in all_tokens_sorted[:10]])
print([(x, tokenizer_orig.convert_ids_to_tokens([x])[0], c) for x,c in all_tokens_sorted[-10:]])
print(len(all_tokens))

print(len(all_data_tokens["dev"] - all_data_tokens["train"]), len(all_data_tokens["train"] - all_data_tokens["dev"]))
print(len(all_data_tokens["test"] - all_data_tokens["train"]), len(all_data_tokens["train"] - all_data_tokens["test"]))

# print([(x, tokenizer_orig.convert_ids_to_tokens([x])[0], all_tokens[x]) for x in all_data_tokens["dev"] - all_data_tokens["train"]])

more_dev_tokens = all_data_tokens["dev"] - all_data_tokens["train"]
more_test_tokens = all_data_tokens["test"] - all_data_tokens["train"]
for k in more_dev_tokens.union(more_test_tokens):
    del all_tokens[k]

print("vocab size 1", len(all_tokens))

vocab = all_tokens.most_common(max_vocab_size-2)
vocab = [tokenizer_orig.unk_token_id, tokenizer_orig.pad_token_id] + sorted([k for k, v in vocab])

print("vocab size", len(vocab))

torch.save(vocab, os.path.join(save_dir, "vocab.pt"))
with open(os.path.join(save_dir, "vocab.txt"), "w") as f:
    f.write("\n".join(tokenizer_orig.convert_ids_to_tokens(vocab)))

tokenizer = ToyTokenizer(model_path, os.path.join(save_dir, "vocab.pt"))
orig2new_vocab_map = {v: k for k, v in enumerate(vocab)}

all_data_tokens_new = {}
for split in all_tokenized_data:
    tokenized_data_new = list(map(lambda x: [orig2new_vocab_map.get(t, 0) for t in x], all_tokenized_data[split]))
    all_data_tokens_new[split] = set([t for d in tokenized_data_new for t in d])
    
    print(tokenized_data_new[0])
    print(tokenizer.decode(tokenized_data_new[0]))
    
    tokenized_data_new = torch.tensor(tokenized_data_new)
    torch.save(tokenized_data_new, os.path.join(save_dir, f"{split}.pt"))
    
print(len(all_data_tokens_new["dev"] - all_data_tokens_new["train"]), len(all_data_tokens_new["train"] - all_data_tokens_new["dev"]))
print(len(all_data_tokens_new["test"] - all_data_tokens_new["train"]), len(all_data_tokens_new["train"] - all_data_tokens_new["test"]))