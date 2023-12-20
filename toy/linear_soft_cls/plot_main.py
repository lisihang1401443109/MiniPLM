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

# for e in range(40):
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))


step = 500
all_dev_losses, all_weighted_ratios, all_areas = [], [], []

cm = plt.colormaps['RdYlBu']

for path in paths:
    weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
    dev_loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    area = sum(dev_loss)
    weighted_ratio = weighted_ratio[:step]
    dev_loss = dev_loss[:step]
    all_dev_losses.append(dev_loss)
    all_weighted_ratios.append(weighted_ratio)
    all_areas.append(area)

all_dev_losses = np.array(all_dev_losses)
all_weighted_ratios = np.array(all_weighted_ratios)
all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_dev_losses.shape[1], axis=1)

# ax[0].scatter(all_dev_losses, all_weighted_ratios, c=all_areas_repeat, cmap=cm)

# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[0].set_xlabel(f"{split}_loss")
# ax[0].set_ylabel("weighted_ratio")
# ax[0].invert_xaxis()


all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = 2000 / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm.reversed()(all_cp_rate_norm)

for dev_loss, weighted_ratio, area_color in zip(all_dev_losses, all_weighted_ratios, all_colors):
    sorted_dev_loss, sorted_weighted_ratio = zip(*sorted(zip(dev_loss, weighted_ratio)))
    # remove < 0.01 values in sorted_weighted_ratios and corresponding values in sorted_dev_loss
    sorted_dev_loss, sorted_weighted_ratio = zip(*[(d, w) for d, w in zip(sorted_dev_loss, sorted_weighted_ratio) if w > 0.1])
    sorted_weighted_ratio = 1 / np.array(sorted_weighted_ratio)
    # 根据 area 的值确定颜色
    ax.plot(sorted_dev_loss, sorted_weighted_ratio, c=area_color)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"{split}_loss")
ax.set_ylabel(r"Std ($\frac{\text{IF}}{\text{Mean IF}}$)")
ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm.reversed(), norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
plt.colorbar(sm, ax=ax)

plt.savefig(os.path.join(base_path, f"{split}_main_inv.png"), bbox_inches="tight")
plt.close()
