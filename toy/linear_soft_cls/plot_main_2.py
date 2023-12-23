import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_epoch_5"), "opt_epoch_5"),
    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]

plot, ax = plt.subplots(3, 1, figsize=(6, 12))

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

cm = plt.colormaps['RdYlBu']

all_IF_ratio, all_loss = [], []

for path in paths:
    weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
    loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    all_loss.append(loss)
    all_IF_ratio.append(weighted_ratio)
    
loss_threshold = all_loss[0][-1]
# loss_threshold = 0
     
print(loss_threshold)
        
all_mean_ratio, all_cp = [], []

max_IF_mean_step = len(all_IF_ratio[0])
for (k,loss), IF_ratio in zip(enumerate(all_loss), all_IF_ratio):
    if k > 0:
        for i in range(len(loss)-1, -1, -1):
            if loss[i] > loss_threshold:
                max_IF_mean_step = i
                break
    
    IF_ratio = IF_ratio[:max_IF_mean_step]
    # loss = loss[:max_IF_mean_step]
    if k >= 80 if split == "test" else 54:
        break
    mean_ratio = np.mean(IF_ratio)
    all_mean_ratio.append(mean_ratio)
    cp = len(loss) / np.sum(loss)
    all_cp.append(cp)
    if k in [0, 1, 8, 9, 10, 11, 12, 104]:
        print(k, mean_ratio)
        ax[1].plot(IF_ratio, label=f"{k}")
        ax[2].plot(loss, label=f"{k}")
    ax[2].hlines(loss_threshold, 0, len(loss), color="red", linestyle="--")
    # # ax[2].set_yscale("log")

ax[2].set_ylim(0.1, 0.15)



idxs = list(range(len(all_cp)))
ax[0].plot(all_mean_ratio, all_cp, marker='o')
# for idx in idxs:
#     ax[0].annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
ax[0].set_xlabel(r"Mean $\frac{\overline{IF}}{Std(IF)}$")
ax[0].set_ylabel(r"Compression Rate")
plt.savefig(os.path.join(base_path, f"cp_vs_mean_ratio_2_{split}.png"))