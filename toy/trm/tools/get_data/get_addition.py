import random
import os
import sys
import torch
import numpy as np

random.seed(42)

base_path = sys.argv[1]
dev_num = 512
train_num = 4096
ratio_1_2 = 1.3

save_dir = os.path.join(base_path, "processed_data", "toy-add-100", f"{train_num}_{dev_num}_{ratio_1_2}")
os.makedirs(save_dir, exist_ok=True)

def get_label(x, y):
    return ((x + y) // 10) % 10

all_data = []
for i in range(100):
    all_data.extend([(i, j, get_label(i,j)) for j in range(100)])

random.shuffle(all_data)
dev_data = all_data[:dev_num]
test_data = all_data[dev_num:2*dev_num]
train_data = all_data[2*dev_num:]

split_1 = [x for x in train_data if x[2] < 5]
split_2 = [x for x in train_data if x[2] >= 5]

if ratio_1_2 > 1:
    split_2 = split_2[:int(len(split_2) / ratio_1_2)]
else:
    split_1 = split_1[:int(len(split_1) * ratio_1_2)]
    
train_data = split_1 + split_2

random.shuffle(train_data)
train_data = train_data[:train_num]

train_data = torch.tensor(train_data, dtype=torch.long)
dev_data = torch.tensor(dev_data, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

print(len(train_data), len(dev_data), len(test_data))

torch.save(train_data, os.path.join(save_dir, "train.pt"))
torch.save(dev_data, os.path.join(save_dir, "dev.pt"))
torch.save(test_data, os.path.join(save_dir, "test.pt"))
