import matplotlib.pyplot as plt
import torch
import os

# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_epoch_5"), "opt_epoch_5"),
    (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    (os.path.join(base_path, "dyna"), "dyna"),
]

plot, ax = plt.subplots(3, 1, figsize=(10, 10))

step = 300

for path in paths:
    var_IF = torch.load(os.path.join(path[0], f"var_IF_{split}.pt"), map_location="cpu")[:step]
    weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")[:step]
    std_IF = torch.load(os.path.join(path[0], f"std_IF_{split}.pt"), map_location="cpu")[:step]
    weighted_mean_IF = torch.load(os.path.join(path[0], f"weighted_mean_IF_{split}.pt"), map_location="cpu")[:step]
    weighted_mean_IF = [-x for x in weighted_mean_IF]
    dev_loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")[:step]
    ax[0].scatter(dev_loss, var_IF, label=path[1])
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].scatter(dev_loss, weighted_ratio, label=path[1])
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[2].scatter(weighted_mean_IF, std_IF, label=path[1])
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    

ax[0].set_xlabel(f"{split}_loss")
ax[0].set_ylabel("var_IF")
ax[0].invert_xaxis()
ax[0].legend()
ax[1].set_xlabel(f"{split}_loss")
ax[1].set_ylabel("weighted_ratio")
ax[1].invert_xaxis()
ax[1].legend()
ax[2].set_xlabel("weighted_mean_IF")
ax[2].set_ylabel("std_IF")
ax[2].legend()

plt.savefig(os.path.join(base_path, f"{split}_scatter.png"))
plt.close()
