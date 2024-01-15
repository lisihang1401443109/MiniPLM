import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1/"

bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/test_loss.pt"), map_location="cpu")
opt_test_losses = torch.load(os.path.join(base_path, f"opt_0.001_epoch_490/test_loss.pt"), map_location="cpu")

max_steps = 2000

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant Policy", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"(Near) Optimal Policy", color="red")
ax1.set_ylabel(r"Target Loss", fontsize=14)
ax1.set_xlabel(r"Training Time Steps $t$", fontsize=14)
# ax1.set_yticks([0.1, 0.15, 0.2])
ax1.set_ylim([0.09, 0.2])
ax1.get_yaxis().get_major_formatter().labelOnlyBase = False
ax1.minorticks_off()

ax1.set_yscale("log")
ax1.tick_params(axis='both', which='both', labelsize=14)

plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.annotate(text='', xy=(max_steps,bsl_test_losses[-1]-0.001), xytext=(390,bsl_test_losses[-1]-0.001), arrowprops=dict(arrowstyle='<->'))
plt.text(1300, bsl_test_losses[-1]-0.008, r"$5.5 \times$", fontsize=14)

plt.legend(fontsize=14)

plt.savefig(os.path.join(base_path, "losses.pdf"), bbox_inches="tight")