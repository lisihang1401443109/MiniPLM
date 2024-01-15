import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e5000/-0.8_30-eval_opt/10-20-7/"

_, bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
_, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.4/15/all_loss.pt"), map_location="cpu")

max_steps = 5000

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant Policy", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"(Near) Optimal Policy", color="red")
ax1.set_ylabel(r"Target Loss", fontsize=14)
ax1.set_xlabel(r"Training Time Steps $t$", fontsize=14)

ax1.set_yscale("log")
ax1.tick_params(axis='both', which='both', labelsize=14)
ax1.set_ylim([3, 5.5])

plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.annotate(text='', xy=(max_steps,bsl_test_losses[-1]), xytext=(1290,bsl_test_losses[-1]), arrowprops=dict(arrowstyle='<->'))
plt.text(2000, bsl_test_losses[-1]-0.17, r"$2.5 \times$", fontsize=14)

plt.legend(fontsize=14)

plt.savefig(os.path.join(base_path, "losses.pdf"), bbox_inches="tight")