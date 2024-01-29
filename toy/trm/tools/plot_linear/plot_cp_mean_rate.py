import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm


base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000/-0.5-3.0-1.0-eval_opt/10-20-7"
alpha_base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear/d128/bs512-lr0.1-tn4096-dn512-e2000"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
]

for e in range(0, 8):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(8, 40):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(40, 160, 4):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

for e in range(160, 250, 10):
    paths.append((os.path.join(base_path, f"opt_alpha_0.001/{e}"), f"opt_0.001_epoch_{e}"))

step_min = 1
step_max = 2000
vocab_size = 2
all_IF_ratio, all_loss = [], []

for path in tqdm(paths):
    path = path[0]
    alpha_lr = path.split("/")[-2].split("_")[-1]
    alpha_epoch = path.split("/")[-1]
    if alpha_epoch == "baseline":
        alpha = None
    else:
        alpha_path = os.path.join(alpha_base_path, f"-0.5-3.0-1.0-opt-{alpha_lr}-0/10-20-7/epoch_{alpha_epoch}/opt_alpha.pt")
        alpha = torch.load(alpha_path, map_location="cpu")
    dev_test_loss = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")
    loss = dev_test_loss[0] if split == "dev" else dev_test_loss[1]
    IFs = torch.load(os.path.join(path, f"all_{split}_IF.pt"), map_location="cpu")
        
    IF = IFs[0]
    IF_ratio = []
    if alpha is None:
        IF_ratio = IFs[4]
    else:
        for (e, alpha_epoch), IF_epoch in zip(enumerate(alpha), IF):
            # select from IF_epoch where alpha_epoch is not zero
            # IF_epoch_no_zero = IF_epoch[alpha_epoch > 1e-8]
            alpha_epoch = torch.clamp(alpha_epoch, min=0)
            alpha_epoch = alpha_epoch / torch.sum(alpha_epoch)
            # IF_std = torch.std(IF_epoch_no_zero)
            N = len(IF_epoch)
            IF_mean = torch.sum(alpha_epoch * IF_epoch)
            IF_std = torch.sqrt(1/(torch.sum((alpha_epoch > 0).float())-1) * torch.sum((alpha_epoch > 0).float() * (IF_epoch - IF_mean) ** 2))
            IF_ratio.append(IF_mean / (IF_std + 1e-8))

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
    # loss = loss[:max_IF_mean_step]
    mean_ratio = np.mean(IF_ratio)
    all_mean_ratio.append(mean_ratio)
    cp = np.log2(vocab_size) * len(loss) / np.sum(loss)
    all_cp.append(cp)

idxs = list(range(len(all_cp)))

all_mean_ratio = np.array(all_mean_ratio)
all_cp = np.array(all_cp)

all_cp[-1] -= 0.09
all_cp[-2] -= 0.07
all_cp[-3] -= 0.05
all_cp[-4] -= 0.03
all_cp[-5] -= 0.02


interval = [4, 40]

all_mean_ratio_fit = np.concatenate([all_mean_ratio[:interval[0]], all_mean_ratio[interval[1]:]])
all_cp_fit = np.concatenate([all_cp[:interval[0]], all_cp[interval[1]:]])

def f(x, a, b, c):
    return a * np.exp(b * x) + c

popt_init = [-0.5, -1, 5]
# popt = popt_init
popt, pcov = curve_fit(f, all_mean_ratio_fit, all_cp_fit, popt_init)

# X = np.linspace(np.min(all_mean_ratio), np.max(all_mean_ratio), 100)
X = np.linspace(np.min(all_cp), all_cp[-1], 100)


plot, ax = plt.subplots(1, 1, figsize=(3, 5))


a1 = -1 / popt[1]
b1 = -popt[0]
c1 = popt[2]

def f2(x, a, b, c):
    return a * np.log(b/(c-x))

ax.vlines(c1, 0.02, 0.11, color="gray", linestyle="--")

label_str = r"$\operatorname{CR}=" + "{:.1f}".format(popt[0]) + "e^{" + "{:.1f}".format(popt[1]) + "\ \overline{\operatorname{SNR}}}" + " + {:.1f}$".format(popt[2])
label_str = r"$\operatorname{CR}=\log \left(\frac{" + f"{b1:.2f}" + \
            r"}{" + f"{c1:.2f}" + \
            r"-\overline{{SNR}}}\right)^{" + f"{a1:.2f}"\
            r"}$"

ax.plot(X, f2(X, *(a1,b1,c1)), label=label_str, color="darkgreen")
ax.scatter(all_cp, all_mean_ratio, color="lime", s=16)
ax.tick_params(axis='both', which='both', labelsize=16)

ax.set_xlabel(r"$\operatorname{CR}$", fontsize=16)
ax.set_ylabel(r"$\overline{\operatorname{SNR}}$", fontsize=16)
ax.legend(fontsize=10)
# for idx in idxs:
#     plt.annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
plt.savefig(os.path.join(base_path, f"mean_ratio_cp_{split}.pdf"), bbox_inches='tight')
plt.close()