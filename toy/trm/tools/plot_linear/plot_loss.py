import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000/-0.5-3.0-1.0-eval_opt/10-20-7/"

_, bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
_, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.001/490/all_loss.pt"), map_location="cpu")

max_steps = 2000

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"(Near) Optimal", color="red")
ax1.set_ylabel(r"Target Loss $L^{\text{tg}}$", fontsize=14)
ax1.set_xlabel(r"Training Time Steps $t$", fontsize=14)
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.set_ylim(0.09, 0.2)
# ax1.yaxis.set_minor_formatter(ScalarFormatter())
# ax1.minorticks_off()
# ax1.set_yticks([0.1, 0.2], ["0.1", "0.2"])

ax1.set_yscale("log")
ax1.tick_params(axis='both', which='both', labelsize=14)

plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.annotate(text='', xy=(max_steps,bsl_test_losses[-1]-0.001), xytext=(390,bsl_test_losses[-1]-0.001), arrowprops=dict(arrowstyle='<->'))
plt.text(1300, bsl_test_losses[-1]-0.008, r"$\operatorname{AR}(T) = 5.5$", fontsize=12)

plt.legend(fontsize=14)

plt.savefig(os.path.join(base_path, "losses.pdf"), bbox_inches="tight")