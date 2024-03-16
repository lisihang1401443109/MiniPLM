import os
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"

max_steps = 4000
std_window = 600

_, bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
# _, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.4/19/all_loss.pt"), map_location="cpu")
_, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.6/15/all_loss.pt"), map_location="cpu")

bsl_std = []
for i, l in enumerate(tqdm(bsl_test_losses)):
    b = i
    e = min(i + std_window, max_steps)
    if i < 50:
        b = 50
    if e == max_steps:
        b = e - std_window
    std = np.std(bsl_test_losses[b:e])
    bsl_std.append(std)
bsl_std = np.array(bsl_std)

opt_std = []
for i, l in enumerate(tqdm(opt_test_losses)):
    b = i
    e = min(i + std_window, max_steps)
    if i < 50:
        b = 50
    if e == max_steps:
        b = e - std_window
    std = np.std(opt_test_losses[b:e])
    opt_std.append(std)
opt_std = np.array(opt_std)

bsl_test_losses = gaussian_filter1d(bsl_test_losses, sigma=100)
opt_test_losses = gaussian_filter1d(opt_test_losses, sigma=100)

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"Near-Optimal", color="red")

ax1.fill_between(range(10,len(bsl_test_losses)+10), bsl_test_losses-bsl_std, bsl_test_losses+bsl_std, facecolor="mediumblue", alpha=0.2)
ax1.fill_between(range(10,len(opt_test_losses)+10), opt_test_losses-opt_std, opt_test_losses+opt_std, facecolor="red", alpha=0.2)

ax1.set_ylabel(r"$L^{\text{dsr}}(\theta_t)$", fontsize=14)
ax1.set_xlabel(r"Training Time Steps $t$", fontsize=14)

ax1.set_yscale("log")
ax1.tick_params(axis='both', which='both', labelsize=14)
ax1.set_ylim([3.1, 5.5])

plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.annotate(text='', xy=(max_steps,bsl_test_losses[-1]-0.01), xytext=(1700,bsl_test_losses[-1]-0.01), arrowprops=dict(arrowstyle='<->'))
plt.text(2600, bsl_test_losses[-1]-0.17, r"$2.41 \times$", fontsize=14)

plt.legend(fontsize=14)
# plt.title("Transformer Language Modeling", fontsize=14)

plt.savefig(os.path.join("/home/lidong1/yuxian/sps-toy/results/toy/icml/", "losses.pdf"), bbox_inches="tight")