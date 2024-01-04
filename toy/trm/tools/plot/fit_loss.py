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

base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e2000/-eval_opt/10-20-7"

paths = [
    os.path.join(base_path, "baseline"),
    # base_path = "/home/aiscuser/sps/results/toy/trm/toy-trm-ts-64/bs512-lr0.1-tn4096-dn512-e10000/10-20-7/baseline"
    os.path.join(base_path, "opt_alpha/9"),
    os.path.join(base_path, "opt_alpha/39"),
    
]



split = "dev"

for path in paths:
    losses = torch.load(os.path.join(path, f"all_loss.pt"), map_location="cpu")

    if split == "dev":
        loss = losses[0]
    else:
        loss = losses[1]

    loss = np.array(loss)
    
    loss = gaussian_filter1d(loss, sigma=1)
    
    steps = np.array(range(len(loss))) + 1

    min_steps = 200

    loss = loss[min_steps:]
    steps = steps[min_steps:]

    bias = 0

    log_loss = np.log(loss - bias)
    log_steps = np.log(steps)
    

    plt.scatter(steps, log_loss, s=5, alpha=0.05)

    popt, pcov = curve_fit(f, log_steps, log_loss)

    # print(popt, pcov)
    a = popt[0]
    b = popt[1] + a * np.log(min_steps)
    # b = popt[1]

    r_2 = compute_r_square(log_steps, log_loss, f, popt)

    print("R^2:", r_2)

    plt.plot(steps, f(log_steps, *popt), "--", label=f"y={a:.5f}x'+{b:.5f}, R^2={r_2:.4f}")

# plt.xticks(log_steps, steps)

plt.xscale("log")
plt.xticks([200, 400, 1000, 1500, 2000], [200, 400, 1000, 1500, 2000])
plt.legend()
plt.savefig(os.path.join(base_path, f"{split}_loss_b{bias}_ms{min_steps}.png"))