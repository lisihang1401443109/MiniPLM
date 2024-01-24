import os
import torch
import numpy as np
from matplotlib import pyplot as plt

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7/"

T = 4000
vocab_size = 5000

test_areas, test_crs = [], []

steps = [-1] + list(range(15))

A0 = T * np.log2(vocab_size)

for i in steps:
    if i == -1:
        _, test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
    else:
        _, test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.4/{i}/all_loss.pt"), map_location="cpu")
    test_area = 1 / T * np.sum(test_losses)
    cr = A0 / np.sum(test_losses)
    
    test_areas.append(test_area)
    test_crs.append(cr)

steps = [s + 1 for s in steps]

fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(steps, test_areas, label=r"Loss: $J(\gamma)$", color="tab:green")
ax1.set_ylabel(r"Target Loss AUC", fontsize=14)
ax2 = ax1.twinx()
l2, = ax2.plot(steps, test_crs, label=r"Compression Rate", linestyle="--", color="tab:green")

l3 = ax1.scatter(steps[-1], test_areas[-1], color="red", marker="*", s=140, label=r"Near-Optimal", zorder=10)
ax2.scatter(steps[-1], test_crs[-1], color="red", marker="*", s=140, zorder=10)
l4 = ax1.scatter(steps[0], test_areas[0], color="mediumblue", marker="s", s=50, label=r"Constant", zorder=10)
ax2.scatter(steps[0], test_crs[0], color="mediumblue", marker="s", s=50, zorder=10)

ax2.set_ylabel(r"Compression Rate ($\operatorname{CR}$)", rotation=-90, labelpad=20, fontsize=14)
ax1.set_xlabel(r"Optimization Epochs", fontsize=14)
ax1.tick_params(axis='both', which='both', labelsize=14)
ax2.tick_params(axis='both', which='both', labelsize=14)

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc="center right", fontsize=10)

plt.savefig(os.path.join(base_path, "area_loss.pdf"), bbox_inches="tight")