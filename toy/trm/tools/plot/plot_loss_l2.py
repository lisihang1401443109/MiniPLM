import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"


_, bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
# _, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.4/19/all_loss.pt"), map_location="cpu")
_, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.6/15/all_loss.pt"), map_location="cpu")

bsl_test_losses = gaussian_filter1d(bsl_test_losses, sigma=100)
opt_test_losses = gaussian_filter1d(opt_test_losses, sigma=100)

max_steps = 4000

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"Near-Optimal", color="red")
ax1.set_ylabel(r"Target Loss $L^{\text{tg}}(\theta_t)$", fontsize=14)
ax1.set_xlabel(r"Training Time Steps $t$", fontsize=14)

ax1.set_yscale("log")
ax1.tick_params(axis='both', which='both', labelsize=14)
ax1.set_ylim([3.1, 5.5])

plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.annotate(text='', xy=(max_steps,bsl_test_losses[-1]-0.01), xytext=(1700,bsl_test_losses[-1]-0.01), arrowprops=dict(arrowstyle='<->'))
plt.text(2600, bsl_test_losses[-1]-0.17, r"$2.41 \times$", fontsize=12)

plt.legend(fontsize=14)

plt.savefig(os.path.join(base_path, "losses.pdf"), bbox_inches="tight")