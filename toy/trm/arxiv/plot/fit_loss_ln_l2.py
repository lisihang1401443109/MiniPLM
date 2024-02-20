from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import os
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def f(x, a, b):
    return a * x + b

def compute_r_square(x, y, f, popt):
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - f(x, *popt)) ** 2)
    return 1 - ss_res / ss_tot


# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"
# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"
# base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"

base_path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/toy-trm-l2-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e4000/-0.8_30-eval_opt/10-20-7"

paths = [
    os.path.join(base_path, "baseline"),
    # os.path.join(base_path, "opt_alpha_0.1/0"),
    # os.path.join(base_path, "opt_alpha_0.1/1"), 
    # os.path.join(base_path, "opt_alpha_0.1/2"),   
    # os.path.join(base_path, "opt_alpha_0.1/3"),
    # os.path.join(base_path, "opt_alpha_0.1/4"), 
    # os.path.join(base_path, "opt_alpha_0.1/5"),
    # os.path.join(base_path, "opt_alpha_0.1/6"),
    # os.path.join(base_path, "opt_alpha_0.1/7"),
    # os.path.join(base_path, "opt_alpha_0.1/8"),
    # os.path.join(base_path, "opt_alpha_0.4/0"),
    # os.path.join(base_path, "opt_alpha_0.4/5"),
    # os.path.join(base_path, "opt_alpha_0.4/10"),
    # os.path.join(base_path, "opt_alpha_0.4/19"),
    # os.path.join(base_path, "opt_alpha_0.5/15"),
    os.path.join(base_path, "opt_alpha_0.6/15"),    
]

min_steps = 400
max_steps = 3400
all_steps = 4000
bias = 0

split = "test"

plot, ax = plt.subplots(1, 1, figsize=(8, 4))

betas, Bs, all_losses = [], [], []

for i, path in enumerate(paths):
    losses = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")

    if split == "dev":
        loss = losses[0]
    else:
        loss = losses[1]

    loss = np.array(loss)
    
    loss = gaussian_filter1d(loss, sigma=100)
    
    all_losses.append(loss)
    
    steps = np.array(range(len(loss))) + 1

    loss = loss[:max_steps]
    steps = steps[:max_steps]

    log_loss = np.log(loss - bias)
    log_steps = np.log(steps)
    
    fit_log_loss = log_loss[min_steps:]
    fit_log_steps = log_steps[min_steps:]

    plt.scatter(steps[min_steps:], loss[min_steps:]-bias, s=15, color="lightblue" if i == 0 else "lightcoral", label="Constant Policy" if i == 0 else "Near-Optimal Policy")
    # plt.scatter(steps, loss, s=5, color="cyan" if i == 0 else "coral", label="Constant Policy" if i == 0 else "(Near) Optimal Policy")

    popt, pcov = curve_fit(f, fit_log_steps, fit_log_loss)

    # print(popt, pcov)
    a = popt[0]
    b = popt[1]

    r_2 = compute_r_square(fit_log_steps, fit_log_loss, f, popt)

    print("R^2:", r_2)

    T_min = np.exp(-b/a)
    print(T_min, np.exp(b), -a)
    T_min_1 = int(np.log10(T_min))
    T_min_2 = T_min / (10 ** T_min_1)
    
    betas.append(-a)
    Bs.append(T_min)
    
    # label_str = r"$\ln(L^{\text{tg}})=" + f"{a:.3f}" + r"\ln(t)+" + f"{b:.2f}" + r", r^2=" + f"{r_2:.3f}$"
    label_str = r"$L^{\text{dsr}}=\left(" + f"{T_min_2:.1f}" + r"E" + f"{T_min_1}" + r"/t\right)^{" + f"{-a:.2f}" + r"}, r^2=" + f"{r_2:.3f}" + r"$"
    ax.plot(steps[min_steps:], np.exp(f(log_steps[min_steps:], *popt)), "--", label=label_str, linewidth=3, color="blue" if i == 0 else "darkred")
    # ax.plot(steps, np.exp(f(log_steps, *popt)), linestyle="dashed", label=label_str, linewidth=1.5, color="blue" if i == 0 else "darkred")

def acc(t, B1, B2, beta1, beta2):
    return (B1**(beta1/beta2)/B2) * (t ** (1-beta1/beta2))

acc_rate = acc(all_steps, Bs[0], Bs[1], betas[0], betas[1])
acc_rate_exp = acc(10e6, Bs[0], Bs[1], betas[0], betas[1])

bsl_loss = all_losses[0][-1]
for i in range(0, len(all_losses[1])):
    if all_losses[1][i] < bsl_loss:
        break

acc_rate_real = all_steps / i

print("acc_rate_real: ", acc_rate_real, "acc_rate: ", acc_rate, "acc_rate_exp: ", acc_rate_exp)
print("improvement: ", (Bs[0]-Bs[1])/Bs[0], (betas[1]-betas[0])/betas[0])

print(acc_rate, acc_rate_exp, (Bs[0]-Bs[1])/Bs[0], (betas[1]-betas[0])/betas[0])

# plt.xticks(log_steps, steps)
ax.set_xlabel(r"$\text{Training Steps} \ t$", fontsize=18)
ax.set_ylabel(r"$L^{\text{dsr}}(\theta_t)$", fontsize=18)
# ax.set_xlabel(r"$\ln(t)$", fontsize=14)
# ax.set_ylabel(r"$\ln(L^{\text{tg}})$", fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
# plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([400, 1000, 2000, 3000], [400, 1000, 2000, 3000])
ax.set_yticks([3.5, 4.0, 4.5, 5.0], [3.5, 4.0, 4.5, 5.0])
ax.tick_params(axis='both', which='both', labelsize=18)
plt.legend(fontsize=15)
# plt.title("Transformer Language Modeling", fontsize=18)
plt.savefig(os.path.join("/home/lidong1/yuxian/sps-toy/results/toy/icml",
            f"{split}_loss_b{bias}_ms{min_steps}.png"), bbox_inches="tight", dpi=300)
