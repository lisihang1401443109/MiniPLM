import os
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000/-0.5-3.0-1.0-eval_opt/10-20-7/"

_, bsl_test_losses = torch.load(os.path.join(base_path, f"baseline/all_loss.pt"), map_location="cpu")
_, opt_test_losses = torch.load(os.path.join(base_path, f"opt_alpha_0.001/490/all_loss.pt"), map_location="cpu")

max_steps = 2000

std_window = 800

bsl_test_losses = bsl_test_losses[:max_steps]
opt_test_losses = opt_test_losses[:max_steps]

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

print(bsl_std)

fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(111)

l1, = ax1.plot(bsl_test_losses, label=r"Constant", color="mediumblue")
l2, = ax1.plot(opt_test_losses, label=r"Near-Optimal", color="red")
ax1.fill_between(range(10, len(bsl_test_losses) + 10), bsl_test_losses-bsl_std, bsl_test_losses+bsl_std, facecolor="mediumblue", alpha=0.2)
ax1.fill_between(range(10, len(opt_test_losses) + 10), opt_test_losses-opt_std, opt_test_losses+opt_std, facecolor="red", alpha=0.2)
ax1.set_ylabel(r"$L^{\text{dsr}}(\theta_t)$", fontsize=14)
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
plt.text(1300, bsl_test_losses[-1]-0.008, r"$5.50 \times$", fontsize=12)

plt.legend(fontsize=14)
plt.title("Perceptron Linear Classification", fontsize=14)

plt.savefig(os.path.join(base_path, "losses.pdf"), bbox_inches="tight")