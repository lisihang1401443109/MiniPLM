import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"
alpha_base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_alpha_0.1/0"), "opt_alpha_0"),
    # (os.path.join(base_path, "opt_alpha_0.1/1"), "opt_alpha_1"),
    # (os.path.join(base_path, "opt_alpha_0.1/2"), "opt_alpha_2"),
    # (os.path.join(base_path, "opt_alpha_0.1/3"), "opt_alpha_3"),
    # (os.path.join(base_path, "opt_alpha_0.1/4"), "opt_alpha_4"),
    # (os.path.join(base_path, "opt_alpha_0.1/5"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.1/6"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.1/7"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/0"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/1"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/2"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/3"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/4"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/5"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.2/10"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/15"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/0"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/1"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/2"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/3"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/4"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/5"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/10"), "opt_alpha_5"),
    # (os.path.join(base_path, "opt_alpha_0.4/15"), "opt_alpha_5"),

]


split = "test"

plot, ax = plt.subplots(1, 1, figsize=(4, 3))


step_min = 0
step_max = 3000
vocab_size = 5000
tot_info = 3000
train_num = 16384
all_alpha_0, all_areas = [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    path = path[0]
    alpha_lr = path.split("/")[-2].split("_")[-1]
    alpha_epoch = path.split("/")[-1]
    if alpha_epoch == "baseline":
        alpha = torch.ones(step_max, train_num) / train_num
    else:
        alpha_path = os.path.join(alpha_base_path, f"-0.8_30-opt-{alpha_lr}-0/10-20-7/epoch_{alpha_epoch}/opt_alpha.pt")
        alpha = torch.load(alpha_path, map_location="cpu")
    
    alpha = torch.clamp(alpha, min=0)
    alpha = alpha / torch.sum(alpha, keepdim=True, dim=-1)

    alpha = alpha / torch.max(alpha, dim=-1, keepdim=True)[0]

    alpha = alpha.numpy()

    all_train_losses_per_inst = torch.load(os.path.join(path, "all_train_losses_per_inst.pt"), map_location="cpu")

    all_train_losses_per_inst = all_train_losses_per_inst.numpy()

    alpha = alpha[step_min:step_max]
    all_train_losses_per_inst = all_train_losses_per_inst[step_min:step_max]

    print(np.min(all_train_losses_per_inst), np.max(all_train_losses_per_inst))

    alpha_0 = alpha[(all_train_losses_per_inst < 10)]

    alpha_0 = alpha_0.reshape(-1)
    alpha_0 = np.concatenate([alpha_0, np.ones(1)+0.1], axis=-1)

    all_loss = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    area = sum(loss)

    all_alpha_0.append(alpha_0)
    all_areas.append(area)

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = tot_info * np.log2(vocab_size) / all_areas
print(all_cp_rate)
all_cp_rate_norm = all_cp_rate - min(all_cp_rate)
all_cp_rate_norm = all_cp_rate_norm / max(all_cp_rate_norm)
all_colors = cm(all_cp_rate_norm)

for alpha_0, color in zip(all_alpha_0, all_colors):
    ax.hist(alpha_0, bins=1000, density=True, color=color, cumulative=True, histtype='step', linewidth=2)

# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$\gamma_n(t)\ /\ \max (\gamma_n(t))$", fontsize=14)
ax.set_ylabel("Cumulative Probility", fontsize=14)
ax.set_xlim(0, 1)
# ax.set_xlabel(r"$L^{\text{tg}}(\mathbf{\theta}_t)$", fontsize=14)
# set the font size of x-axis and y-axis
ax.tick_params(axis='both', which='both', labelsize=14)
# ax.set_ylim(ymin=0)
# ax.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(all_cp_rate), vmax=max(all_cp_rate)))
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label(r"$\operatorname{CR}$", fontsize=14, labelpad=-40, y=-0.04, rotation=0)

plt.savefig(os.path.join(base_path, f"{split}_loss_alpha.png"), bbox_inches="tight")
plt.close()