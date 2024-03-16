import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"

alpha_base_path = "/home/lidong1/yuxian/sps-toy//results/toy/opt_alpha/0.5-3.0-1.0-4096-10-20-1-d128-ns2000-na4096-eta0.1-lr0.001"

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

# for e in range(40):
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))

min_step = 1
step = 1000
all_dev_losses, all_weighted_ratios, all_areas = [], [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    dev_loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    area = sum(dev_loss)
    dev_loss = dev_loss[min_step:step]
    
    IF = torch.load(os.path.join(path[0], f"IF_{split}.pt"), map_location="cpu")
    IF = torch.stack(IF, dim=0).squeeze()
    IF = -IF
    policy = path[0].split("/")[-1]
    if policy == "baseline":
        weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
        weighted_ratio = weighted_ratio[:step]
    else:
        alpha_epoch = int(policy.split("_")[-1])
        alpha_path = os.path.join(alpha_base_path, f"epoch_{alpha_epoch}", "opt_alpha.pt")
        alpha = torch.load(alpha_path, map_location="cpu")
        alpha = torch.clamp(alpha, min=0)
        alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        IF_mean = torch.sum(alpha * IF, dim=-1, keepdim=True)
        mask = (alpha > 0).float()
        N = torch.sum(mask,dim=-1)
        a = 1/(N-1)
        d = (IF - IF_mean) ** 2
        IF_std = torch.sqrt(a * torch.sum(mask * d, dim=-1))
        weighted_ratio = IF_mean.squeeze() / (IF_std + 1e-8)
        weighted_ratio = weighted_ratio.tolist()
    weighted_ratio = weighted_ratio[min_step:step]
    
    all_dev_losses.append(dev_loss)
    all_areas.append(area)
    all_weighted_ratios.append(weighted_ratio)

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
# print(all_areas)
all_cp_rate = 2000 / all_areas
# print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm(all_cp_rate_norm)

for (i, dev_loss), weighted_ratio, area_color in zip(enumerate(all_dev_losses), all_weighted_ratios, all_colors):
    sorted_dev_loss, sorted_weighted_ratio = zip(*sorted(zip(dev_loss, weighted_ratio)))
    # print(len(sorted_dev_loss), len(sorted_weighted_ratio))
    # print(sorted_weighted_ratio)
    # remove < 0.01 values in sorted_weighted_ratios and corresponding values in sorted_dev_loss
    sorted_dev_loss, sorted_weighted_ratio = zip(*[(d, w) for d, w in zip(sorted_dev_loss, sorted_weighted_ratio)])
    sorted_weighted_ratio = np.array(sorted_weighted_ratio)
    # 根据 area 的值确定颜色
    ax.plot(sorted_dev_loss, sorted_weighted_ratio, c=area_color)

ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$L^{\text{tg}}(\mathbf{\theta}_t)$", fontsize=14)
ax.set_ylabel(r"$\operatorname{SNR}(t)$", fontsize=14)
# set the font size of x-axis and y-axis
ax.tick_params(axis='both', which='both', labelsize=14)
ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_ticks(np.arange(4, 7, 0.5))
cbar.set_label(r"$\operatorname{CR}$", fontsize=14, labelpad=-30, y=-0.04, rotation=0)

plt.savefig(os.path.join(base_path, f"{split}_main.pdf"), bbox_inches="tight")
plt.close()
