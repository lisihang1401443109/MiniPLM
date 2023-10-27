import matplotlib.pyplot as plt
import re
import os
import pickle

paths = [
    ("/home/lidong1/yuxian/sps/results/fairseq/pretrain/fairseq_125M/t500K-w10K-bs8-lr0.0003cosine3e-05-G2-N16-NN2-scr/plot", "125M", "losses"),
    ("/home/lidong1/yuxian/sps/results/fairseq/pretrain/fairseq_218M/t500K-w10K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/plot", "218M", "losses"),
    ("/home/lidong1/yuxian/sps/results/fairseq/pretrain/fairseq_250M/t500K-w10K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/plot", "250M", "losses"),
    ("/home/lidong1/yuxian/sps/results/fairseq/pretrain/fairseq_250M-2/t500K-w10K-bs8-lr0.0003cosine3e-05-G4-N8-NN1-scr/plot", "250M-2", "losses"),
    ("/home/lidong1/yuxian/sps/results/fairseq/pretrain/fairseq_448M/t500K-w10K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/plot", "448M", "losses"),
    ("/home/lidong1/yuxian/sps/results/fairseq/kd_rsd/fairseq_355M/t500K-w10K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/fairseq_1.3B-fairseq_125M-kd0.5/plot", "kd-125-355-1.3", "lm_losses")
]

max_step = 50000
min_step = 20000
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