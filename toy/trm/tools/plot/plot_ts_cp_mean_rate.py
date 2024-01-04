import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm


base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e2000/-eval_opt/10-20-7"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
]

# for e in range(0, 400, 10):
#     paths.append((os.path.join(base_path, f"opt_alpha_{e}_wm700"), f"opt_alpha_{e}_wm700"))

# for e in range(40):
#     paths.append((os.path.join(base_path, f"opt_alpha/{e}"), f"opt_alpha_{e}"))

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    (os.path.join(base_path, "opt_alpha_0.1/0"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/1"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/2"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/3"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/4"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/5"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha_0.1/6"), "opt_alpha_0.1_0"),
    (os.path.join(base_path, "opt_alpha/0"), "opt_alpha_0"),
    (os.path.join(base_path, "opt_alpha/1"), "opt_alpha_1"),
    (os.path.join(base_path, "opt_alpha/2"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/3"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/4"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/5"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/6"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/7"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/8"), "opt_alpha_2"),
    (os.path.join(base_path, "opt_alpha/9"), "opt_alpha_2"),    
    (os.path.join(base_path, "opt_alpha/10"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/11"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/12"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/13"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/14"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/15"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/16"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/17"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/18"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/19"), "opt_alpha_10"),
    (os.path.join(base_path, "opt_alpha/20"), "opt_alpha_20"),
    (os.path.join(base_path, "opt_alpha/30"), "opt_alpha_30"),
    (os.path.join(base_path, "opt_alpha/39"), "opt_alpha_39"),
    
    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]

for i in range(11, 20):
    paths.append((os.path.join(base_path, f"opt_alpha_0.1/{i}"), f"opt_alpha_0.1_{i}"))

plot, ax = plt.subplots(1, 1, figsize=(10, 5))


step_min = 500
step_max = 2000
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
    cp = np.log(4000) * len(loss) / np.sum(loss)
    all_cp.append(cp)

idxs = list(range(len(all_cp)))

all_mean_ratio = np.array(all_mean_ratio)
all_cp = np.array(all_cp)

plt.scatter(all_mean_ratio, all_cp)


def f(x, a, b, c, d):
    return c*(1/(1 + np.exp(-a*(x-b)))-0.5) + d

d1 = np.mean(all_cp)
all_cp_scale = (all_cp - d1)
c1 = 1 / np.max(all_cp_scale)
all_cp_scale = c1 * all_cp_scale

b1 = np.mean(all_mean_ratio)
all_mean_ratio_scale = (all_mean_ratio - b1)
a1 = 1 / np.max(all_mean_ratio_scale)
all_mean_ratio_scale = a1 * all_mean_ratio_scale
# popt = [10, 0, 1, 0]
popt, pcov = curve_fit(f, all_mean_ratio_scale, all_cp_scale, [10, 0, 1, 0])

a,b,c,d = popt

popt_orig = [a1*a, (a1*b1+b)/a1, c/c1, d/c1+d1]

X = np.linspace(np.min(all_mean_ratio), np.max(all_mean_ratio), 100)

plt.plot(X, f(X, *popt_orig))

plt.xlabel("mean IF ratio")
plt.ylabel("Compression rate")
# for idx in idxs:
#     plt.annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
plt.savefig(os.path.join(base_path, f"mean_ratio_cp_{split}.png"))
plt.close()