import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"
alpha_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.4-0/10-20-7/epoch_5/opt_alpha.pt"

split = "test"

plot, ax = plt.subplots(1, 1, figsize=(6, 3))


step_min = 0
step_max = 200
vocab_size = 2
tot_info = 2000
all_areas, all_rate_alpha_no_zero = [], []

cm = plt.colormaps['coolwarm']

alpha = torch.load(alpha_path, map_location="cpu")
alpha = torch.clamp(alpha, min=0, max=torch.max(alpha)/1000)
alpha = alpha / torch.sum(alpha, keepdim=True, dim=-1)


alpha = alpha.numpy()

sample_num = 3000

# randomly sample 3000 rows from alpha
idx = np.random.choice(alpha.shape[1], sample_num, replace=False)
alpha = alpha[:, idx]

alpha = alpha.T
# sort alpha rows by the sum of each row
alpha = alpha[np.argsort(np.sum(alpha, axis=1))]
# alpha = alpha[np.argsort(alpha[:, 0])]


print(alpha.shape)

max_alpha = np.max(alpha)
min_alpha = np.min(alpha)

# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$L^{\text{tg}}(\mathbf{\theta}_t)$", fontsize=14)
ax.set_ylabel(r"$\operatorname{SNR}(t)$", fontsize=14)
# set the font size of x-axis and y-axis
ax.tick_params(axis='both', which='both', labelsize=14)
# ax.set_ylim(ymin=0)
# ax.invert_xaxis()
print(alpha.shape)
ax.pcolormesh(alpha, cmap=cm, vmin=min_alpha, vmax=max_alpha)

sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min_alpha, vmax=max_alpha))
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label(r"$\gamma$", fontsize=14, labelpad=-40, y=-0.04, rotation=0)

plt.savefig(os.path.join(base_path, f"{split}_alpha.png"), bbox_inches="tight")
plt.close()