import torch
import random
import os
from copy import deepcopy
from tqdm import tqdm
from tiny_story_model import ToyTokenizer

seed = 42

random.seed(seed)
torch.manual_seed(seed)

noise_fraction = 0.5
N = 20
max_length = 64
rept_times = 10000

base_path = "/home/aiscuser/sps/processed_data/toy-ts/mistral/small_64_16384_512_2"
model_path = "/mnt/yuxian/checkpoints/mistral-7B/"
vocab = torch.load(os.path.join(base_path, "vocab.pt"))
torch.load(os.path.join(base_path, "train.pt"))

tokenizer = ToyTokenizer(model_path, os.path.join(base_path, "vocab.pt"))

def repeat(x):
    idx = random.randint(0, len(x)-1)
    x.insert(idx, x[idx])
    return x

def delete(x):
    x = x[:-1]
    return x

def replace(x):
    rep = random.randint(0, len(vocab)-1)
    idx = random.randint(0, len(x)-1)
    x[idx] = rep
    return x

def repeat_sample(data):
    for _ in range(rept_times):
        idx1 = random.randint(0, len(data)-1)
        idx2 = idx1
        while idx2 == idx1:
            idx2 = random.randint(0, len(data)-1)
        data[idx1] = deepcopy(data[idx2])
    return data


train_data = torch.load(os.path.join(base_path, "train.pt")).tolist()

noise_data = train_data[:int(len(train_data)*noise_fraction)]

new_noise_data = []

print(tokenizer.decode(noise_data[0]))

for x in tqdm(noise_data):
    for _ in range(N):
        x = random.choice([repeat, delete, replace])(x)
    x = x[:max_length+1]
    x = x + [tokenizer.pad_token_id] * (max_length+1 - len(x))
    new_noise_data.append(x)

print(tokenizer.decode(new_noise_data[0]))

new_train_data = new_noise_data + train_data[int(len(train_data)*noise_fraction):]

random.shuffle(new_train_data)

new_train_data = repeat_sample(new_train_data)

new_train_data = torch.tensor(new_train_data)

save_path = os.path.join(base_path, f"noise_train_{noise_fraction}_{N}_{rept_times}.pt")

torch.save(new_train_data, save_path)