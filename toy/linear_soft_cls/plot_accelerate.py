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

# for e in [25]:
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))


def calc_acc_rate(baseline_losses, losses, steps=None):
    def _calc(step):
        if step >= len(baseline_losses):
            return -1
        baseline_loss = baseline_losses[step]
        # binary search baseline_loss in losses
        l, r = 0, len(losses) - 1
        while l < r:
            mid = (l + r) // 2
            if losses[mid] >= baseline_loss:
                l = mid + 1
            else:
                r = mid
        return step / l

    acc_rate = [round(_calc(step), 3) for step in steps]

    return acc_rate


n = 20
acc_steps = [int(i * 1000 / n) for i in range(4, n)]
all_losses = []
all_areas = []

cm = plt.colormaps['RdYlBu']

for path in paths:
    weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
    losses = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    area = sum(losses)
    all_losses.append(losses)
    all_areas.append(area)

all_losses = np.array(all_losses)

all_acc_rates = []
for losses in all_losses:
    acc_rate = calc_acc_rate(all_losses[0], losses, acc_steps)
    all_acc_rates.append(acc_rate)

all_acc_rates = np.array(all_acc_rates)

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = 2000 / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm.reversed()(all_cp_rate_norm)

for acc_rate, area_color in zip(all_acc_rates, all_colors):
    # 根据 area 的值确定颜色
    ax.plot(acc_steps, acc_rate, c=area_color)

# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(f"steps")
ax.set_ylabel("Acceleration Rate")
# ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm.reversed(), norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
plt.colorbar(sm, ax=ax)

plt.savefig(os.path.join(base_path, f"{split}_acc.png"), bbox_inches="tight")
plt.close()
