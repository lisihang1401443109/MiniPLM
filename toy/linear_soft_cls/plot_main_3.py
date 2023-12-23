import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_epoch_5"), "opt_epoch_5"),
    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]

plot, ax = plt.subplots(1, 1, figsize=(6, 6))

for e in range(0, 8):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(8, 40):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(40, 160, 4):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(160, 500, 10):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

# for e in [1, 4, 12, 40, 80, 160, 390, 490]:
#     paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

# for e in range(40):
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))


step = 500
all_mean_ratio, all_cp = [], []

cm = plt.colormaps['RdYlBu']

for path in paths:
    weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
    dev_loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    area = sum(dev_loss)
    cp = 2000 / area
    all_cp.append(cp)
    mean_ratio = np.mean(weighted_ratio)
    all_mean_ratio.append(mean_ratio)
    
plt.plot(all_mean_ratio, all_cp, marker='o')
plt.xlabel(r"Mean $\frac{\overline{IF}}{Std(IF)}$")
plt.ylabel(r"Compression Rate")
plt.savefig(os.path.join(base_path, "cp_vs_mean_ratio.png"))