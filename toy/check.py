import torch
import os
from matplotlib import pyplot as plt


# path = "/home/lidong1/yuxian/sps-toy/results/toy/linear-d128-l0.1/bs-1-lr0.001/_oe5_lra1e-3_dev_noise/"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/linear-d128-l0.1/bs-1-lr0.001/oe5-lra0.001-dmu0.5-dsig0.1-dnoi0.01"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/linear-d128-l0.1/bs-1-lr0.001-tn2048-dn256/oe5-lra0.001-dmu0.5-dsig0.1-dnoi0.01"
path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_da/d128-None-l0.0/bs-1-lr0.001-tn2048-dn256"

# best_alpha = torch.load(os.path.join(path, "best_alpha.pt"), map_location="cpu").squeeze()

# print(best_alpha)

# sorted_alpha = torch.sort(best_alpha, descending=True)[0]

# # 画柱状图
# plt.bar(range(sorted_alpha.size(0)), sorted_alpha)
# plt.savefig(os.path.join(path, "best_alpha.png"))
# plt.close()


# naive_best_alpha = torch.load(os.path.join(path, "naive_best_alpha.pt"), map_location="cpu").squeeze()

# print(naive_best_alpha)

# naive_sorted_alpha = torch.sort(naive_best_alpha, descending=True)[0]

# # 画柱状图
# plt.bar(range(naive_sorted_alpha.size(0)), naive_sorted_alpha)
# plt.savefig(os.path.join(path, "naive_best_alpha.png"))
# plt.close()

alpha_10 = torch.load(os.path.join(path, "alpha-10.pt"), map_location="cpu").squeeze()
alpha_100 = torch.load(os.path.join(path, "alpha-100.pt"), map_location="cpu").squeeze()
alpha_1000 = torch.load(os.path.join(path, "alpha-1000.pt"), map_location="cpu").squeeze()

sorted_alpha_10, indices = torch.sort(alpha_10, descending=True)
sorted_alpha_100 = alpha_100[indices]
sorted_alpha_1000 = alpha_1000[indices]

plt.plot(range(sorted_alpha_10.size(0)), sorted_alpha_10, label="alpha-10")
plt.plot(range(sorted_alpha_100.size(0)), sorted_alpha_100, label="alpha-100")
plt.plot(range(sorted_alpha_1000.size(0)), sorted_alpha_1000, label="alpha-1000")

plt.legend()
plt.savefig(os.path.join(path, "alpha_cmp.png"))
plt.close()