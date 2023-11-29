import torch
import os
from matplotlib import pyplot as plt


path = "/home/lidong1/yuxian/sps-toy/results/toy/linear-d128-l0.1/bs-1-lr0.01/_iter5_lra1e-3/"

best_alpha = torch.load(os.path.join(path, "best_alpha.pt"), map_location="cpu").squeeze()

print(best_alpha)

sorted_alpha = torch.sort(best_alpha, descending=True)[0]

# 画柱状图
plt.bar(range(sorted_alpha.size(0)), sorted_alpha)
plt.savefig(os.path.join(path, "alpha.png"))