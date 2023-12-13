import matplotlib.pyplot as plt
import torch
import os

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_results_toy_opt_alpha_d128-ns2000-na1024-eta0.1-lr0.1_epoch_9"), "epoch9"),
    (os.path.join(base_path, "opt_results_toy_opt_alpha_d128-ns2000-na1024-eta0.1-lr0.1_epoch_39"), "epoch39"),
    
]

plot, ax = plt.subplots(2, 1, figsize=(10, 10))

for path in paths:
    var_IF = torch.load(os.path.join(path[0], "var_IF.pt"), map_location="cpu")
    weighted_ratio = torch.load(os.path.join(path[0], "weighted_ratio.pt"), map_location="cpu")
    dev_loss = torch.load(os.path.join(path[0], "dev_loss.pt"), map_location="cpu")
    dev_loss = dev_loss
    ax[0].scatter(dev_loss, var_IF, label=path[1])
    ax[1].scatter(dev_loss, weighted_ratio, label=path[1])

ax[0].set_xlabel("dev_loss")
ax[0].set_ylabel("var_IF")
ax[0].invert_xaxis()
ax[0].legend()
ax[1].set_xlabel("dev_loss")
ax[1].set_ylabel("weighted_ratio")
ax[1].invert_xaxis()
ax[1].legend()

plt.savefig(os.path.join(base_path, "scatter.png"))
plt.close()
