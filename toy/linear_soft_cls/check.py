import torch
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import numpy as np

# path_dyna = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj/"

# dyna_alpha = torch.load(os.path.join(path_dyna, "all_alpha.pt"), map_location="cpu")

# dyna_alpha = dyna_alpha.numpy()

# plt.figure(figsize=(10, 10))
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# for e in [0, 39]:
#     path_opt = "/home/lidong1/yuxian/sps-toy/results/toy/opt_alpha/d128-ns2000-na1024-eta0.1-lr0.1/epoch_39/"

#     opt_alpha = torch.load(os.path.join(path_opt, "opt_alpha.pt"), map_location="cpu").numpy()

#     print(dyna_alpha.shape)
#     print(opt_alpha.shape)

#     all_p_corr = []
#     all_s_corr = []

#     for i in range(dyna_alpha.shape[0]):
#         p_corr = pearsonr(dyna_alpha[i], opt_alpha[i])[0]
#         s_corr = spearmanr(dyna_alpha[i], opt_alpha[i])[0]
#         all_p_corr.append(p_corr)
#         all_s_corr.append(s_corr)
    
#     ax[0].plot(all_p_corr, label=f"epoch_{e}")
#     ax[1].plot(all_s_corr, label=f"epoch_{e}")

# ax[0].legend()
# ax[1].legend()
# plt.savefig(os.path.join(path_dyna, "corr.png"))  
# plt.close()

# def plot_alpha_dist(alpha, name):
#     alpha_0 = alpha[0]
#     alpha_10 = alpha[10]
#     alpha_100 = alpha[100]
#     alpha_1000 = alpha[1000]

#     sorted_alpha_0, alpha_idx = torch.sort(alpha_0, descending=True)
#     sorted_alpha_10 = alpha_10[alpha_idx]
#     sorted_alpha_100 = alpha_100[alpha_idx]
#     sorted_alpha_1000 = alpha_1000[alpha_idx]

#     # plt.figure(figsize=(10, 10))

#     plt.plot(range(len(sorted_alpha_0)), sorted_alpha_0.numpy(), label="epoch_0")
#     plt.plot(range(len(sorted_alpha_10)), sorted_alpha_10.numpy(), label="epoch_10")
#     plt.plot(range(len(sorted_alpha_100)), sorted_alpha_100.numpy(), label="epoch_100")
#     plt.plot(range(len(sorted_alpha_1000)), sorted_alpha_1000.numpy(), label="epoch_1000")
#     plt.legend()
#     plt.savefig(os.path.join(path_dyna, f"{name}.png"))
#     plt.close()
    
# plot_alpha_dist(dyna_alpha, "dyna_alpha")

# path_opt = "/home/lidong1/yuxian/sps-toy/results/toy/opt_alpha/d128-ns2000-na1024-eta0.1-lr0.1/epoch_36/"
# opt_alpha = torch.load(os.path.join(path_opt, "opt_alpha.pt"), map_location="cpu")

# plot_alpha_dist(opt_alpha, "opt_alpha")

# def plot_alpha_change(alpha, path):
#     sorted_alpha_0, alpha_idx = torch.sort(alpha[0], descending=True)

#     plt.plot(range(len(sorted_alpha_0)), sorted_alpha_0.numpy(), label="epoch_0")
#     plt.legend()
#     plt.savefig(os.path.join(path, f"alpha_0.png"))
#     plt.close()

#     for idx in [0, 10, 100, 1000]:
#         s_idx = alpha_idx[idx]
#         alpha_change = [alpha[t][s_idx].item() for t in range(alpha.shape[0])]
#         alpha_change = alpha_change[:200]
#         plt.plot(range(len(alpha_change)), alpha_change, label=f"alpha_{idx}")
#     plt.legend()
#     plt.savefig(os.path.join(path, f"alpha_change.png"))
#     plt.close()

# # plot_alpha_change(dyna_alpha, path_dyna)
# plot_alpha_change(opt_alpha, path_opt)

base_if_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn4096-dn512/lra0.0004-tmu0.0-tsig3.0-dmu0.5-dsig1.0-aui1-proj/10-20-1/"

plot, ax = plt.subplots(2, 1, figsize=(10, 10))

for name in ["baseline", "opt_epoch_35"]:
    IF = torch.load(
        os.path.join(base_if_path, name, "IF.pt"), map_location="cpu")

    e = 100

    print(torch.mean(IF[e]), torch.var(IF[e]))
    zero_num = (torch.abs(IF[e]) < 1e-3).sum()
    print("origin shape", IF[e].shape)
    print("zero_num", zero_num)

    no_zero_IF = IF[e][torch.abs(IF[e]) > 0.3 * torch.std(IF[e])]

    print("new_shape", no_zero_IF.shape)

    stat, p = stats.shapiro(no_zero_IF)
    print("test normal", stat, p)
    p = stats.normaltest(no_zero_IF)
    print("test normal", p)

    print(torch.mean(no_zero_IF), torch.var(no_zero_IF))

    ax[0].hist(IF[e].numpy(), bins=50, label=f"{name}")
    ax[0].set_ylim(0, 300)
    ax[0].set_title("IF")
    ax[1].hist(no_zero_IF.numpy(), bins=50, label=f"{name}")
    ax[1].set_ylim(0, 300)
    ax[1].set_title("IF without zero")

plt.legend()
plt.savefig(os.path.join(base_if_path, name, f"IF_{e}.png"))
plt.close()

# base_if_path = "/home/lidong1/yuxian/sps-toy/results/toy/linear_soft_cls_da/d128-None-l0.0/bs-1-lr0.1-tn1024-dn512/lra0.0004-tmu0.0-tsig2.0-dmu0.5-dsig2.0-aui1-proj/"

# IF_opt = torch.load(
#     os.path.join(base_if_path, "IF_opt_results_toy_opt_alpha_d128-ns2000-na1024-eta0.1-lr0.1_epoch_9.pt"), map_location="cpu")

# e = 0

# print(IF_opt[e])
# stat, p = stats.shapiro(IF_opt[e])
# print(stat, p)
# p = stats.normaltest(IF_opt[e])
# print(p)

# plt.hist(IF_opt[e].numpy(), bins=50, density=True)
# plt.savefig(os.path.join(base_if_path, f"IF_opt_e{e}.png"))
# plt.close()
