import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"
alpha_path = "/home/lidong1/yuxian/sps-toy/results/toy/opt_alpha/0.5-3.0-1.0-4096-10-20-1-d128-ns2000-na4096-eta0.1-lr0.001/epoch_900/opt_alpha.pt"

split = "test"

plot, ax = plt.subplots(1, 1, figsize=(6, 3))


step_min = 0
step_max = 200
all_areas, all_rate_alpha_no_zero = [], []

cm = plt.colormaps['coolwarm']

alpha = torch.load(alpha_path, map_location="cpu")[step_min:step_max]

no_zero_nums = torch.sum(alpha > 5e-4, dim=-1)

print(no_zero_nums[:100])

alpha = torch.clamp(alpha, min=0, max=torch.max(alpha)/5)
alpha = alpha / torch.sum(alpha, keepdim=True, dim=-1)


alpha = alpha.numpy()

sample_num = 4096

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