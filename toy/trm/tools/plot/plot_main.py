import os
import torch
import numpy as np
import matplotlib.pyplot as plt


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/10-20-7"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_alpha_0"), "opt_alpha_0"),
    (os.path.join(base_path, "opt_alpha_1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha_2"), "opt_alpha_2"),

    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]

plot, ax = plt.subplots(1, 1, figsize=(6, 3))


step_min = 500
step_max = 2000
all_losses, all_IF_ratios, all_areas = [], [], []

cm = plt.colormaps['RdYlBu']

for path in paths:
    all_loss = torch.load(os.path.join(path[0], f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    IF_mean, IF_var, IF_std, IF_ratio = torch.load(os.path.join(path[0], f"all_{split}_IF.pt"), map_location="cpu")
    area = sum(loss)
    IF_ratio = IF_ratio[step_min:step_max]
    loss = loss[step_min:step_max]
    all_losses.append(loss)
    all_IF_ratios.append(IF_ratio)
    all_areas.append(area)

all_losses = np.array(all_losses)
all_IF_ratios = np.array(all_IF_ratios)
all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_losses.shape[1], axis=1)

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = 2000 / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm.reversed()(all_cp_rate_norm)

for loss, IF_ratio, area_color in zip(all_losses, all_IF_ratios, all_colors):
    sorted_loss, sorted_IF_ratio = zip(*sorted(zip(loss, IF_ratio)))
    # remove < 0.01 values in sorted_IF_ratios and corresponding values in sorted_dev_loss
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio) if w > 0.1])
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio) if d > 0.5 and d < 1.2])
    
    # sorted_IF_ratio = 1 / np.array(sorted_IF_ratio)
    # 根据 area 的值确定颜色
    ax.plot(sorted_loss, sorted_IF_ratio, c=area_color)

# ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"{split}_loss")
ax.set_ylabel(r"Std ($\frac{\text{IF}}{\text{Mean IF}}$)")
ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm.reversed(), norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
plt.colorbar(sm, ax=ax)

plt.savefig(os.path.join(base_path, f"{split}_main.png"), bbox_inches="tight")
plt.close()
