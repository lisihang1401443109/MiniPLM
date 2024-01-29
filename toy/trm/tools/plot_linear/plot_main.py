import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000/-0.5-3.0-1.0-eval_opt/10-20-7"
alpha_base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_alpha_0.001/0"), "opt_alpha_0"),
    # (os.path.join(base_path, "opt_alpha_0.001/1"), "opt_alpha_1"),
    # (os.path.join(base_path, "opt_alpha_0.001/2"), "opt_alpha_2"),
    # (os.path.join(base_path, "opt_alpha_0.001/490"), "opt_alpha_490"),
]

for e in range(0, 8):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(8, 40):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(40, 160, 4):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(160, 300, 10):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

plot, ax = plt.subplots(1, 1, figsize=(5, 3))


step_min = 1
step_max = 1000
vocab_size = 2
tot_info = 2000
all_losses, all_IF_ratios, all_areas = [], [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    path = path[0]
    alpha_lr = path.split("/")[-2].split("_")[-1]
    alpha_epoch = path.split("/")[-1]
    if alpha_epoch == "baseline":
        alpha = None
    else:
        alpha_path = os.path.join(alpha_base_path, f"-0.5-3.0-1.0-opt-{alpha_lr}-0/10-20-7/epoch_{alpha_epoch}/opt_alpha.pt")
        alpha = torch.load(alpha_path, map_location="cpu")
    all_loss = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    IFs = torch.load(os.path.join(path, f"all_{split}_IF.pt"), map_location="cpu")
        
    IF = IFs[0]
    IF_ratio = []
    if alpha is None:
        IF_ratio = IFs[4]
    else:
        for (e, alpha_epoch), IF_epoch in zip(enumerate(alpha), IF):
            # select from IF_epoch where alpha_epoch is not zero
            # IF_epoch_no_zero = IF_epoch[alpha_epoch > 1e-8]
            alpha_epoch = torch.clamp(alpha_epoch, min=0)
            alpha_epoch = alpha_epoch / torch.sum(alpha_epoch)
            # IF_std = torch.std(IF_epoch_no_zero)
            N = len(IF_epoch)
            IF_mean = torch.sum(alpha_epoch * IF_epoch)
            IF_std = torch.sqrt(1/(torch.sum((alpha_epoch > 0).float())-1) * torch.sum((alpha_epoch > 0).float() * (IF_epoch - IF_mean) ** 2))
            IF_ratio.append(IF_mean / (IF_std + 1e-8))
    # IF_ratio = IFs[4]
    area = sum(loss)
    IF_ratio = IF_ratio[step_min:step_max]
    loss = loss[step_min:step_max]
    all_losses.append(loss)
    all_IF_ratios.append(IF_ratio)
    all_areas.append(area)

# all_losses = [gaussian_filter1d(loss, sigma=1) for loss in all_losses]
# all_IF_ratios = [gaussian_filter1d(IF_ratio, sigma=100) for IF_ratio in all_IF_ratios]

all_losses = np.array(all_losses)
all_IF_ratios = np.array(all_IF_ratios)

all_areas_repeat = np.repeat(np.expand_dims(all_areas, axis=1), all_losses.shape[1], axis=1)

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = tot_info * np.log2(vocab_size) / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm(all_cp_rate_norm)

for loss, IF_ratio, area_color in zip(all_losses, all_IF_ratios, all_colors):
    sorted_loss, sorted_IF_ratio = zip(*sorted(zip(loss, IF_ratio), reverse=True))
    # remove < 0.01 values in sorted_IF_ratios and corresponding values in sorted_dev_loss
    sorted_loss, sorted_IF_ratio = zip(*[(d, w) for d, w in zip(sorted_loss, sorted_IF_ratio)])
    
    # sorted_loss = gaussian_filter1d(sorted_loss, sigma=1)
    sorted_IF_ratio = gaussian_filter1d(sorted_IF_ratio, sigma=5)
    
    # sorted_IF_ratio = 1 / np.array(sorted_IF_ratio)
    # 根据 area 的值确定颜色
    ax.plot(sorted_loss, sorted_IF_ratio, c=area_color)

ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$L^{\text{dsr}}(\theta_t)$", fontsize=14)
ax.set_ylabel(r"$\operatorname{SNR}_t$", fontsize=14)
# set the font size of x-axis and y-axis
ax.tick_params(axis='both', which='both', labelsize=14)
ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label(r"$\operatorname{CR}$", fontsize=14, labelpad=-20, y=-0.04, rotation=0)

plt.title("Perceptron Linear Classification", fontsize=14)
plt.savefig(os.path.join(base_path, f"{split}_main.pdf"), bbox_inches="tight")
plt.close()