from matplotlib import pyplot as plt
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

base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-eval_opt/10-20-7"

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
    os.path.join(base_path, "opt_alpha_0.4/10"),
    # os.path.join(base_path, "opt_alpha_0.4/15"),
    
]

min_steps = 200
bias = 0

split = "dev"

plt.figure(figsize=(8, 6))

for path in paths:
    losses = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")

    if split == "dev":
        loss = losses[0]
    else:
        loss = losses[1]

    loss = np.array(loss)
    
    loss = gaussian_filter1d(loss, sigma=1)
    
    steps = np.array(range(len(loss))) + 1

    loss = loss[min_steps:]
    steps = steps[min_steps:]

    log_loss = np.log(loss - bias)
    log_steps = np.log(steps)
    
    plt.scatter(steps, log_loss, s=1, alpha=0.05)

    popt, pcov = curve_fit(f, log_steps, log_loss)

    # print(popt, pcov)
    a = popt[0]
    b = popt[1] + a * np.log(min_steps)
    # b = popt[1]

    r_2 = compute_r_square(log_steps, log_loss, f, popt)

    print("R^2:", r_2)

    plt.plot(steps, f(log_steps, *popt), "--", label=f"y={a:.5f}x'+{b:.5f}, R^2={r_2:.4f}", linewidth=0.8)

# plt.xticks(log_steps, steps)
plt.xlabel("ln(Steps)")
plt.ylabel("ln(Loss)")
plt.xscale("log")
plt.xticks([200, 400, 1000, 1500, 2000], [200, 400, 1000, 1500, 2000])
plt.legend()
plt.savefig(os.path.join(base_path, f"{split}_loss_b{bias}_ms{min_steps}.png"))