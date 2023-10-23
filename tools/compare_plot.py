import matplotlib.pyplot as plt
import re
import os
import pickle

paths = [
    ("/home/aiscuser/sps/results/gpt2/train/pretrain/fairseq_125M/t500K-bs8-lr0.0003cosine3e-05-G2-N16-NN2-scr/plot", "pretrain", "losses"),
    ("/home/aiscuser/sps/results/gpt2/train/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd1.0/plot", "rsd1.0", "total_lm_losses"),
]

max_step = 20000
min_step = 1000
smooth = 8

for path, name, key in paths:
    with open(os.path.join(path, "data.pkl"), "rb") as f:
        steps, d = pickle.load(f)
    
    start, end = 0, len(steps)
    for i, step in enumerate(steps):
        if step >= min_step:
            start = i
            break
    for i, step in enumerate(steps):
        if step >= max_step:
            end = i
            break
    
    print(start, end)
    steps = steps[start:end]
    values = d[key][start:end]
    
    smoothed_steps, smoothed_values = [], []
    
    for i in range(len(steps) - smooth):
        smoothed_values.append(sum(values[i:i+smooth]) / len(values[i:i+smooth]))
        smoothed_steps.append(steps[i])

    plt.plot(smoothed_steps, smoothed_values, label=name)
    
os.makedirs("results/plots/compare", exist_ok=True)

all_names = "_".join([name for _, name, _ in paths])

plt.legend()
plt.savefig(f"results/plots/compare/{all_names}.png")
plt.close()