import os
import torch
from collections import Counter

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.005-tn4000-dn500/10-20"
path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.1-tn4000-dn500/10-20"


train_data, dev_data, test_data = torch.load(os.path.join(path, "data.pt"))

print(train_data.size(), dev_data.size(), test_data.size())

c_train = Counter(train_data[:, 2].tolist())
c_dev = Counter(dev_data[:, 2].tolist())
c_test = Counter(test_data[:, 2].tolist())

print(c_train)
print(c_dev)
print(c_test)