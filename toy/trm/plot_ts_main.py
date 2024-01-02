import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e2000/-eval_opt/10-20-7"
base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-5k-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-eval_opt/10-20-7"


split = "dev"

# paths = [
#     (os.path.join(base_path, "baseline"), "baseline"),
#     # (os.path.join(base_path, "opt_alpha_0.1/0"), "opt_alpha_0.1_0"),
#     # (os.path.join(base_path, "opt_alpha_0.1/1"), "opt_alpha_0.1_0"),
#     (os.path.join(base_path, "opt_alpha/0"), "opt_alpha_0"),
#     # (os.path.join(base_path, "opt_alpha/1"), "opt_alpha_1"),
#     (os.path.join(base_path, "opt_alpha/2"), "opt_alpha_2"),
#     # (os.path.join(base_path, "opt_alpha/3"), "opt_alpha_2"),
#     (os.path.join(base_path, "opt_alpha/10"), "opt_alpha_10"),
#     # (os.path.join(base_path, "opt_alpha/5"), "opt_alpha_2"),
#     (os.path.join(base_path, "opt_alpha/20"), "opt_alpha_20"),
#     (os.path.join(base_path, "opt_alpha/30"), "opt_alpha_30"),
#     # (os.path.join(base_path, "opt_alpha/8"), "opt_alpha_2"),
#     (os.path.join(base_path, "opt_alpha/39"), "opt_alpha_39"),    
#     # (os.path.join(base_path, "opt_alpha/10"), "opt_alpha_10"),
#     # (os.path.join(base_path, "opt_alpha/20"), "opt_alpha_20"),
#     # (os.path.join(base_path, "opt_alpha/30"), "opt_alpha_30"),
#     # (os.path.join(base_path, "opt_alpha/39"), "opt_alpha_39"),
    
#     # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
#     # (os.path.join(base_path, "dyna"), "dyna"),
# ]

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_alpha_0.6/0"), "opt_alpha_0"),
    (os.path.join(base_path, "opt_alpha_0.6/1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha_0.6/2"), "opt_alpha_2"),
    # (os.path.join(base_path, "opt_alpha_0.6/3"), "opt_alpha_3"),
    (os.path.join(base_path, "opt_alpha_0.6/5"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.6/10"), "opt_alpha_10"),
    # (os.path.join(base_path, "opt_alpha_0.6/11"), "opt_alpha_11"),
    # (os.path.join(base_path, "opt_alpha_0.6/12"), "opt_alpha_12"),
    # (os.path.join(base_path, "opt_alpha_0.6/11"), "opt_alpha_11"),
    # (os.path.join(base_path, "opt_alpha_0.6/20"), "opt_alpha_20"),
    # (os.path.join(base_path, "opt_alpha_0.6/30"), "opt_alpha_30"),
    # (os.path.join(base_path, "opt_alpha_0.6/39"), "opt_alpha_39"),    
]

plot, ax = plt.subplots(1, 1, figsize=(12, 8))


step_min = 750
step_max = 3000
vocab_size = 5000
tot_info = 3000
all_losses, all_IF_ratios, all_areas = [], [], []

cm = plt.colormaps['RdYlBu']

for path in paths:
    all_loss = torch.load(os.path.join(path[0], f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    IFs = torch.load(os.path.join(path[0], f"all_{split}_IF.pt"), map_location="cpu")
    avg_IF_ratio = IFs[6]
    area = sum(loss)
    avg_IF_ratio = avg_IF_ratio[step_min:step_max]
    loss = loss[step_min:step_max]
    all_losses.append(loss)
    all_IF_ratios.append(avg_IF_ratio)
    all_areas.append(area)

# all_losses = [gaussian_filter1d(loss, sigma=1) for loss in all_losses]
# all_IF_ratios = [gaussian_filter1d(IF_ratio, sigma=100) for IF_ratio in all_IF_ratios]

all_losses = np.array(all_losses)
all_IF_ratios = np.array(all_IF_ratios)


all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_losses.shape[1], axis=1)

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = tot_info * np.log(vocab_size) / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm.reversed()(all_cp_rate_norm)

for loss, IF_ratio, area_color in zip(all_losses, all_IF_ratios, all_colors):
    sorted_loss, sorted_IF_ratio = zip(*sorted(zip(loss, IF_ratio)))
    # remove < 0.01 values in sorted_IF_ratios and corresponding values in sorted_dev_loss
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio) if d > 3.2])
    
    sorted_loss = gaussian_filter1d(sorted_loss, sigma=1)
    sorted_IF_ratio = gaussian_filter1d(sorted_IF_ratio, sigma=100)
    
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
