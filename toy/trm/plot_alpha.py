import torch
from matplotlib import pyplot as plt
import os


base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-5k-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-opt-0.6-0/10-20-7/"

paths = [
    os.path.join(base_path, "epoch_0"),
    os.path.join(base_path, "epoch_5"),
    os.path.join(base_path, "epoch_10"),
    os.path.join(base_path, "epoch_20"),
    
]


steps = [0, 200, 1000, 2000]

for path in paths:
    plot, ax = plt.subplots(2, 1, figsize=(12, 24))
    alphas = torch.load(os.path.join(path, "opt_alpha.pt"), map_location="cpu")
    # _, idxs = torch.sort(alphas[0], descending=True)
    for step in steps:
        # plot_alpha = alphas[step][idxs]
        plot_alpha = torch.sort(alphas[step], descending=True)[0]
        print(torch.sum(plot_alpha))
        ax[0].plot(plot_alpha, label=f"step {step}")

    alpha_idxs = [0, 10, 20, 30, 40, 50]

    alphas = torch.load(os.path.join(path, "opt_alpha.pt"), map_location="cpu")
    for idx in alpha_idxs:
        plot_alpha = alphas[:, idx]
        ax[1].plot(plot_alpha, label=f"alpha {idx}")


    plt.legend()
    plt.savefig(os.path.join(base_path, f"alpha_{path.split('/')[-1]}.png"))
    
    