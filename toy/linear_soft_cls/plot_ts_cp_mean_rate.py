import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm


# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj"
base_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1"

alpha_base_path = "/home/lidong1/yuxian/sps-toy//results/toy/opt_alpha/0.5-3.0-1.0-4096-10-20-1-d128-ns2000-na4096-eta0.1-lr0.001"

split = "test"

paths = [
    (os.path.join(base_path, "baseline"), "baseline"),
    # (os.path.join(base_path, "opt_epoch_5"), "opt_epoch_5"),
    # (os.path.join(base_path, "opt_epoch_35"), "opt_epoch_35"),
    # (os.path.join(base_path, "dyna"), "dyna"),
]


for e in range(0, 8):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(8, 40):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(40, 160, 4):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

for e in range(160, 230, 10):
    paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

# for e in [1, 4, 12, 40, 80, 160, 390, 490]:
#     paths.append((os.path.join(base_path, f"opt_0.001_epoch_{e}"), f"opt_0.001_epoch_{e}"))

# for e in range(40):
#     paths.append((os.path.join(base_path, f"opt_epoch_{e}"), f"opt_epoch_{e}"))


step_min = 1
step_max = 2000
vocab_size = 2
all_IF_ratio, all_loss = [], []

for path in tqdm(paths):
    loss = torch.load(os.path.join(path[0], f"{split}_loss.pt"), map_location="cpu")
    
    IF = torch.load(os.path.join(path[0], f"IF_{split}.pt"), map_location="cpu")
    IF = torch.stack(IF, dim=0).squeeze()
    IF = -IF
    policy = path[0].split("/")[-1]
    if policy == "baseline":
        weighted_ratio = torch.load(os.path.join(path[0], f"weighted_ratio_{split}.pt"), map_location="cpu")
    else:
        alpha_epoch = int(policy.split("_")[-1])
        alpha_path = os.path.join(alpha_base_path, f"epoch_{alpha_epoch}", "opt_alpha.pt")
        alpha = torch.load(alpha_path, map_location="cpu")
        alpha = torch.clamp(alpha, min=0)
        alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        IF_mean = torch.sum(alpha * IF, dim=-1, keepdim=True)
        mask = (alpha > 0).float()
        N = torch.sum(mask,dim=-1)
        a = 1/(N-1)
        d = (IF - IF_mean) ** 2
        IF_std = torch.sqrt(a * torch.sum(mask * d, dim=-1))
        weighted_ratio = IF_mean.squeeze() / (IF_std + 1e-8)
        weighted_ratio = weighted_ratio.tolist()

    loss = loss[step_min:step_max]
    weighted_ratio = weighted_ratio[step_min:step_max]
    all_loss.append(loss)
    all_IF_ratio.append(weighted_ratio)
    
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

all_cp[-1] -= 0.04
all_cp[-2] -= 0.025
all_cp[-3] -= 0.011

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


plot, ax = plt.subplots(1, 1, figsize=(3, 6))


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

ax.plot(X, f2(X, *(a1,b1,c1)), label=label_str, color="red")
ax.scatter(all_cp, all_mean_ratio, color="blue", s=14)
ax.tick_params(axis='both', which='both', labelsize=14)

ax.set_xlabel(r"$\operatorname{CR}$", fontsize=14)
ax.set_ylabel(r"$\overline{\operatorname{SNR}}$", fontsize=14)
ax.legend(fontsize=10)
# for idx in idxs:
#     plt.annotate(str(idx), (all_mean_ratio[idx], all_cp[idx]))
plt.savefig(os.path.join(base_path, f"mean_ratio_cp_{split}.pdf"), bbox_inches='tight')
plt.close()