import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"
alpha_base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_alpha_0.1/0"), "opt_alpha_0"),
    (os.path.join(base_path, "opt_alpha_0.1/1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha_0.1/2"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha_0.1/3"), "opt_alpha_3"),
    (os.path.join(base_path, "opt_alpha_0.1/4"), "opt_alpha_4"),
    (os.path.join(base_path, "opt_alpha_0.1/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.1/6"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.1/7"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/0"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/10"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/15"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/0"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/1"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/2"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/3"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/4"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/10"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/15"), "opt_alpha_5"),
]

plot, ax = plt.subplots(1, 1, figsize=(6, 3))


step_min = 0
step_max = 3000
vocab_size = 5000
tot_info = 3000
all_losses, all_IF_ratios, all_areas = [], [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    path = path[0]
    all_loss = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    
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
print(all_areas)
all_cp_rate = tot_info * np.log(vocab_size) / all_areas
print(all_cp_rate)
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
