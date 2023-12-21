import os
import torch
from collections import Counter

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.005-tn4000-dn500/10-20"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.1-tn4000-dn500/10-20"


# train_data, dev_data, test_data = torch.load(os.path.join(path, "data.pt"))

# print(train_data.size(), dev_data.size(), test_data.size())

# c_train = Counter(train_data[:, 2].tolist())
# c_dev = Counter(dev_data[:, 2].tolist())
# c_test = Counter(test_data[:, 2].tolist())

# print(c_train)
# print(c_dev)
# print(c_test)


p1 = torch.tensor([0.1, 0.2, 0.7])
p2 = torch.tensor([0.15, 0.25, 0.6])
p = torch.tensor([0.3, 0.3, 0.4])

torch.manual_seed(0)
print("p1", torch.multinomial(p1, 1))
for _ in range(10):
    print("p after p1", torch.multinomial(p, 1))

torch.manual_seed(0)
print("p2", torch.multinomial(p2, 1))
for _ in range(10):
    print("p after p2", torch.multinomial(p, 1))
