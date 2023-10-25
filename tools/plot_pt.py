import matplotlib.pyplot as plt
import re
import os
import pickle

# path = "/home/aiscuser/sps/results/gpt2/train/pretrain/fairseq_125M/t500K-bs8-lr0.0003cosine3e-05-G2-N16-NN2-scr/"
# path = "/home/aiscuser/sps/results/fairseq/train/pretrain/fairseq_250M/t500K-bs4-lr0.0003cosine3e-05-G2-N32-NN4-scr/"
path = "/home/aiscuser/sps/results/fairseq/pretrain/fairseq_218M/t500K-bs4-lr0.0003cosine3e-05-G2-N32-NN4-scr/"

with open(os.path.join(path, "log.txt")) as f:
    lines = f.readlines()

r = r"train.*global_steps (\d+)/.* \| loss: (.*) \| elasped_time: .*"

steps = []

d = {
    "losses": []
}

for line in lines:
    m = re.match(r, line)
    if m is not None:
        steps.append(int(m.group(1)))
        d["losses"].append(float(m.group(2)))

print(steps[:10])
print(d["losses"][:10])

plot_save_path = os.path.join(path, "plot")
os.makedirs(plot_save_path, exist_ok=True)

print(plot_save_path)

for k in d:
    plt.plot(steps, d[k])
    plt.savefig(os.path.join(plot_save_path, f"{k}.png"))
    plt.close()

with open(os.path.join(plot_save_path, "data.pkl"), "wb") as f:
    pickle.dump((steps, d), f)
