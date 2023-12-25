import os
import torch
from collections import Counter
import matplotlib.pyplot as plt

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

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/30-20-7/baseline/"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/30-20-7/opt_alpha_330_wm700"

# all_dev_IF = torch.load(os.path.join(path, "all_dev_IF.pt"), map_location="cpu")

# dev_IF = all_dev_IF[0]

# e = 1400

# data_path = "/home/lidong1/yuxian/sps-toy/processed_data/toy-add/tn4000-dn500-r1.3/30-20/data.pt"

# data = torch.load(data_path, map_location="cpu")


# train_labels = data[0][:, 2]

# # print(dev_IF[0].size())

# # exit(0)


# plt.hist(dev_IF[e], bins=4000)

# plt.plot(dev_IF[e], train_labels, "o")

# plt.savefig(os.path.join(path, f"dev_IF_bsl_{e}.png"))

# vocab = torch.load("/mnt/yuxian/data/tinystories/all_data/all_tokens.pt", map_location="cpu")

# print(len(vocab))

g_opt = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_params_opt.pt", map_location="cpu")
grad = torch.load("/home/lidong1/yuxian/sps-toy/toy/grad.pt", map_location="cpu")

delta = g_opt - grad

print(torch.sum(torch.abs(delta)))
