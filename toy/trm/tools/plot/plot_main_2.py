import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/10-20-7"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_alpha_0"), "opt_alpha_0"),
    # (os.path.join(base_path, "opt_alpha_1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha_35"), "opt_alpha_35"),

    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]

plot, ax = plt.subplots(1, 1, figsize=(10, 5))


step_min = 450
step_max = 7000
all_losses, all_IF_ratios, all_areas = [], [], []
all_n_steps = []

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
    all_n_steps.append(len(loss))

al_losses = [gaussian_filter1d(loss, sigma=40) for loss in all_losses]
all_IF_ratios = [gaussian_filter1d(IF_ratio, sigma=10) for IF_ratio in all_IF_ratios]

n_min_steps = min([len(x) for x in al_losses])

def scale_metric(metric):
    scaled_metric = []
    for m in metric:
        n_steps = len(m)
        step_idxs = np.array(list(range(n_steps)))
        if n_steps > n_min_steps:
            step_idxs = step_idxs / n_steps * n_min_steps
            step_idxs = step_idxs.astype(int)
        scaled_m = [[] for _ in range(n_min_steps)]
        for idx in step_idxs:
            scaled_m[idx].append(m[idx])
        scaled_m = [np.mean(x) for x in scaled_m]
        scaled_metric.append(scaled_m)
    
    return scaled_metric

# for loss in all_losses:
# all_losses = np.array(all_losses)
# all_IF_ratios = np.array(all_IF_ratios)
# all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_losses.shape[1], axis=1)

all_areas = np.array(all_areas)
all_n_steps = np.array(all_n_steps)
print(all_areas)
all_cp_rate = 2000 / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm.reversed()(all_cp_rate_norm)



for loss, IF_ratio, area_color in zip(all_losses, all_IF_ratios, all_colors):
    sorted_loss, sorted_IF_ratio = zip(*sorted(zip(loss, IF_ratio)))
    # remove < 0.01 values in sorted_IF_ratios and corresponding values in sorted_dev_loss
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio) if w > 0.05])
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio) if d > 0.1])
    
    # sorted_IF_ratio = 1 / np.array(sorted_IF_ratio)
    # 根据 area 的值确定颜色
    ax.plot(sorted_loss, sorted_IF_ratio, c=area_color)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"{split}_loss")
ax.set_ylabel(r"Std ($\frac{\text{IF}}{\text{Mean IF}}$)")
ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm.reversed(), norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
plt.colorbar(sm, ax=ax)

plt.savefig(os.path.join(base_path, f"{split}_main.png"), bbox_inches="tight")
plt.close()
