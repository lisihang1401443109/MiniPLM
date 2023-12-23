import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/30-20-7"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
]

# for e in range(0, 400, 10):
#     paths.append((os.path.join(base_path, f"opt_alpha_{e}_wm700"), f"opt_alpha_{e}_wm700"))

for e in range(40):
    paths.append((os.path.join(base_path, f"opt_alpha/{e}"), f"opt_alpha_{e}"))

plot, ax = plt.subplots(1, 1, figsize=(10, 5))


step_min = 0
step_max = 2000
all_IF_ratio, all_loss = [], []

for path in paths:
    dev_test_loss = torch.load(os.path.join(path[0], f"all_loss.pt"), map_location="cpu")
    loss = dev_test_loss[0] if split == "dev" else dev_test_loss[1]
    data = torch.load(os.path.join(path[0], f"all_{split}_IF.pt"), map_location="cpu")
    if len(data) == 4:
        IF_mean, IF_var, IF_std, IF_ratio = data
    else:
        _, IF_mean, IF_var, IF_std, IF_ratio = data
    loss = loss[step_min:step_max]
    IF_ratio = IF_ratio[step_min:step_max]
    all_loss.append(loss)
    all_IF_ratio.append(IF_ratio)
    
loss_threshold = all_loss[0][-1]
# loss_threshold = 0

all_mean_ratio, all_cp = [], []

max_IF_mean_step = len(all_IF_ratio[0])
for (k, loss), IF_ratio in zip(enumerate(all_loss), all_IF_ratio):
    if k > 0:
        for i in range(len(loss)-1, -1, -1):
            if loss[i] > loss_threshold:
                max_IF_mean_step = i
                break
    print(max_IF_mean_step)
    IF_ratio = IF_ratio[:max_IF_mean_step]
    loss = loss[:max_IF_mean_step]
    mean_ratio = np.mean(IF_ratio)
    all_mean_ratio.append(mean_ratio)
    cp = np.log(12) * len(loss) / np.sum(loss)
    all_cp.append(cp)

idxs = list(range(len(all_cp)))
plt.plot(all_mean_ratio, all_cp, marker="o")
# for idx in idxs:
#     plt.annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
plt.savefig(os.path.join(base_path, f"mean_ratio_cp_{split}.png"))
plt.close()