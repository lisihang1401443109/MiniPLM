import os
from tqdm import tqdm
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

plot, ax = plt.subplots(1, 1, figsize=(6, 3))

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

# for e in [25]:
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))


step_min = 0
step_max = 3000
vocab_size = 2
tot_info = 3000
all_losses, all_IF_ratios, all_areas = [], [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    path = path[0]
    loss = torch.load(os.path.join(path, f"{split}_loss.pt"), map_location="cpu")
    
    area = sum(loss)
    loss = loss[step_min:step_max]
    all_losses.append(loss)
    all_areas.append(area)

all_losses = np.array(all_losses)
all_IF_ratios = np.array(all_IF_ratios)

bsl_loss = all_losses[0]

def binary_search(loss_w_time, target):
    left = 0
    right = len(loss_w_time) - 1
    while left < right:
        mid = (left + right) // 2
        if loss_w_time[mid][1] > target:
            left = mid + 1
        else:
            right = mid
    return left

all_acc_rate = []
for losses in all_losses:
    losses = [[i, l] for i, l in enumerate(losses)]
    losses = sorted(losses, key=lambda x: x[1], reverse=True)
    min_step = losses[-1][0]
    for k in range(len(losses)-1, -1, -1):
        min_step = min(min_step, losses[k][0])
        losses[k][0] = min_step

    acc_rate = []
    
    # print(losses[:10])
    
    for t, l in enumerate(bsl_loss):
        mid = binary_search(losses, l)
        tt, ll = losses[mid]
        acc_rate.append((t+1) / (tt+1))
    
    # print(acc_rate[:10])
    
    all_acc_rate.append(acc_rate)

all_acc_rate = np.array(all_acc_rate)

all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_losses.shape[1], axis=1)

all_areas = np.array(all_areas)
all_cp_rate = tot_info * np.log(vocab_size) / all_areas
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm(all_cp_rate_norm)

for acc_rate, area_color in zip(all_acc_rate, all_colors):    
    # 根据 area 的值确定颜色
    ax.plot(acc_rate, c=area_color)

# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(f"Time Step $t$")
ax.set_ylabel(r"$\operatorname{AR}(t)$")
# ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
plt.colorbar(sm, ax=ax)

plt.savefig(os.path.join(base_path, f"{split}_acc_rate.pdf"), bbox_inches="tight")
plt.close()
