import matplotlib.pyplot as plt
import re
import os
import pickle

paths = [
    ("/home/aiscuser/sps/results/fairseq/pretrain/fairseq_125M/t500K-bs8-lr0.0003cosine3e-05-G2-N16-NN2-scr/plot", "pt-125M", "losses"),
    ("/home/aiscuser/sps/results/fairseq/pretrain/fairseq_250M/t500K-bs4-lr0.0003cosine3e-05-G2-N32-NN4-scr/plot", "pt-250M", "losses"),
    ("/home/aiscuser/sps/results/fairseq/pretrain/fairseq_218M/t500K-bs4-lr0.0003cosine3e-05-G2-N32-NN4-scr/plot", "pt-218M", "losses"),
    # ("/home/aiscuser/sps/results/fairseq/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd1.0/plot", "rsd1.0-2x125M", "total_lm_losses"),
    # ("/home/aiscuser/sps/results/fairseq/train/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd2.0/plot", "rsd2.0-2x125M", "total_lm_losses"),
    ("/home/aiscuser/sps/results/fairseq/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd0.0/plot", "rsd0.0-2x125M", "total_lm_losses"),
    ("/home/aiscuser/sps/results/fairseq/pt_rsd/fairseq_125M/t500K-w12K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd0.0/plot", "rsd0.0-2x125M-w12K", "total_lm_losses"),
    ("/home/aiscuser/sps/results/fairseq/pt_rsd/fairseq_125M/t500K-w15K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd0.0/plot", "rsd0.0-2x125M-w15K", "total_lm_losses"),
    ("/home/aiscuser/sps/results/fairseq/pt_rsd/fairseq_76M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd0.0-num4/plot", "rsd0.0-4x76M", "total_lm_losses"),
]

max_step = 30000
min_step = 1000
smooth = 32

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

    plt.plot(smoothed_steps, smoothed_values, label=name, linewidth=0.5)
    
os.makedirs("results/plots/compare", exist_ok=True)

all_names = "_".join([name for _, name, _ in paths])

plt.legend()
plt.savefig(f"results/plots/compare/{all_names}.pdf", format="pdf")
plt.close()