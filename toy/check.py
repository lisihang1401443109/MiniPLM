import torch
import os
from matplotlib import pyplot as plt


path = "/home/lidong1/yuxian/sps-toy/results/toy/linear-d128-l0.1/bs-1-lr0.001/_oe5_lra1e-3_dev_noise/"

best_alpha = torch.load(os.path.join(path, "best_alpha.pt"), map_location="cpu").squeeze()

print(best_alpha)

sorted_alpha = torch.sort(best_alpha, descending=True)[0]

# 画柱状图
plt.bar(range(sorted_alpha.size(0)), sorted_alpha)
plt.savefig(os.path.join(path, "best_alpha.png"))
plt.close()


naive_best_alpha = torch.load(os.path.join(path, "naive_best_alpha.pt"), map_location="cpu").squeeze()

print(naive_best_alpha)

naive_sorted_alpha = torch.sort(naive_best_alpha, descending=True)[0]

# 画柱状图
plt.bar(range(naive_sorted_alpha.size(0)), naive_sorted_alpha)
plt.savefig(os.path.join(path, "naive_best_alpha.png"))
plt.close()