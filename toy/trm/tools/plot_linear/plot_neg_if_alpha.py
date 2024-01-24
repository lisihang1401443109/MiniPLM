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
    # (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_alpha_0.001/0"), "opt_alpha_0"),
    # (os.path.join(base_path, "opt_alpha_0.001/1"), "opt_alpha_1"),
    # (os.path.join(base_path, "opt_alpha_0.001/2"), "opt_alpha_2"),
    # (os.path.join(base_path, "opt_alpha_0.001/490"), "opt_alpha_490"),
]

for e in range(0, 8, 2):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(10, 40, 10):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

# for e in range(40, 160, 4):
#     paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

# for e in range(160, 300, 10):
#     paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(40, 500, 20):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

plot, ax = plt.subplots(1, 1, figsize=(3, 3))


step_min = 1
step_max = 10
vocab_size = 2
tot_info = 2000

all_neg_IF_alpha_0_ratio, all_areas = [], []

cm = plt.colormaps['coolwarm']

for path in tqdm(paths):
    path = path[0]
    alpha_lr = path.split("/")[-2].split("_")[-1]
    alpha_epoch = path.split("/")[-1]
    alpha_path = os.path.join(alpha_base_path, f"-0.5-3.0-1.0-opt-{alpha_lr}-0/10-20-7/epoch_{alpha_epoch}/opt_alpha.pt")
    alpha = torch.load(alpha_path, map_location="cpu").numpy()
    all_loss = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")
    loss = all_loss[0] if split == "dev" else all_loss[1]
    IFs = torch.load(os.path.join(path, f"all_{split}_IF.pt"), map_location="cpu")
        
    IF = np.array(IFs[0])
    
    IF = IF[:step_max]
    alpha = alpha[:step_max]
    
    alpha_neg_IF = alpha[IF < 0]
    alpha_neg_IF = alpha_neg_IF.reshape(-1)
    num_alpha_neg_IF_0 = np.sum((alpha_neg_IF < 1e-4))
    all_neg_IF_alpha_0_ratio.append(num_alpha_neg_IF_0 / alpha_neg_IF.shape[0])
    area = sum(loss)
    all_areas.append(area)

# all_losses = [gaussian_filter1d(loss, sigma=1) for loss in all_losses]
# all_IF_ratios = [gaussian_filter1d(IF_ratio, sigma=100) for IF_ratio in all_IF_ratios]

all_areas = np.array(all_areas)
print(all_areas)
all_cp_rate = tot_info * np.log2(vocab_size) / all_areas
print(all_cp_rate)

# sort all_cp_rate and all_neg_IF_alpha_0_ratio
all_cp_rate = np.array(all_cp_rate)
all_neg_IF_alpha_0_ratio = np.array(all_neg_IF_alpha_0_ratio)
sort_idx = np.argsort(all_cp_rate)
all_cp_rate = all_cp_rate[sort_idx]
all_neg_IF_alpha_0_ratio = all_neg_IF_alpha_0_ratio[sort_idx]

ax.plot(all_cp_rate, all_neg_IF_alpha_0_ratio * 100, marker="o", color="blue")

# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$\operatorname{CR}$", fontsize=14)
ax.set_ylabel(r"Fraction of $\gamma_{n,t}=0$ (%)", fontsize=14)
# set the font size of x-axis and y-axis
ax.tick_params(axis='both', which='both', labelsize=14)
# ax.invert_xaxis()

plt.savefig(os.path.join(base_path, f"{split}_neg_if.pdf"), bbox_inches="tight")
plt.close()