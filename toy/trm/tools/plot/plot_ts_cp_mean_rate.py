import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm


base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"

split = "dev"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
]

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_alpha_0.1/0"), "opt_alpha_0"),
    (os.path.join(base_path, "opt_alpha_0.1/1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha_0.1/2"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha_0.1/3"), "opt_alpha_3"),
    (os.path.join(base_path, "opt_alpha_0.1/4"), "opt_alpha_4"),
    (os.path.join(base_path, "opt_alpha_0.1/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.1/6"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.1/7"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/0"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/10"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.2/15"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/0"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/1"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/2"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/3"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/4"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/5"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/10"), "opt_alpha_5"),
    (os.path.join(base_path, "opt_alpha_0.4/15"), "opt_alpha_5"),
    
]

plot, ax = plt.subplots(1, 1, figsize=(5, 4))

step_min = 0
step_max = 2500
vocab_size = 5000
all_IF_ratio, all_loss = [], []

for path in tqdm(paths):
    dev_test_loss = torch.load(os.path.join(path[0], f"all_loss.pt"), map_location="cpu")
    loss = dev_test_loss[0] if split == "dev" else dev_test_loss[1]
    data = torch.load(os.path.join(path[0], f"all_{split}_IF.pt"), map_location="cpu")
    avg_IF_ratio = data[-1]
    loss = loss[step_min:step_max]
    avg_IF_ratio = avg_IF_ratio[step_min:step_max]
    all_loss.append(loss)
    all_IF_ratio.append(avg_IF_ratio)
    
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
    # loss = loss[:max_IF_mean_step]
    mean_ratio = np.mean(IF_ratio)
    all_mean_ratio.append(mean_ratio)
    cp = np.log(vocab_size) * len(loss) / np.sum(loss)
    all_cp.append(cp)

idxs = list(range(len(all_cp)))

all_mean_ratio = np.array(all_mean_ratio)
all_cp = np.array(all_cp)

plt.scatter(all_mean_ratio, all_cp)

def f(x, a, b, c):
    return b * np.exp(a * x) + c

popt_init = [-1, -0.5, 2.3]
popt, pcov = curve_fit(f, all_mean_ratio, all_cp, popt_init)

X = np.linspace(np.min(all_mean_ratio), np.max(all_mean_ratio), 100)

plt.plot(X, f(X, *popt), label=f"y={popt[1]:.4f}e^({popt[0]:.4f}x)+{popt[2]:.4f}")

plt.xlabel("mean IF ratio")
plt.ylabel("Compression rate")
plt.legend()
# for idx in idxs:
#     plt.annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
plt.savefig(os.path.join(base_path, f"mean_ratio_cp_{split}.png"))
plt.close()