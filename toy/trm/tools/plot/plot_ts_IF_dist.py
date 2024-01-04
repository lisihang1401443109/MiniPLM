import torch
import os
from matplotlib import pyplot as plt
from scipy import stats



base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e2000/-eval_opt/10-20-7/baseline"
# base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e2000/-eval_opt/10-20-7/opt_alpha/9"

split = "dev"

d = torch.load(os.path.join(base_path, f"all_{split}_IF.pt"), map_location="cpu")

IF = d[0]

plot, ax = plt.subplots(2, 5, figsize=(12, 6))

for n, e in enumerate([1, 10, 50, 100, 200, 500, 800, 1000, 1500, 1999]):
    i = n // 5
    j = n % 5
    ax[i][j].hist(IF[e], bins=500)
    ax[i][j].set_title(f"step {e}")
    p1 = stats.shapiro(IF[e])
    p2 = stats.normaltest(IF[e])
    print(e, p1, p2)

plt.savefig(os.path.join(base_path, f"{split}_IF.png"))