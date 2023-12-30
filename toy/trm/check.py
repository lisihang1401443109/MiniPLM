import os
import torch
from collections import Counter
import matplotlib.pyplot as plt

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.005-tn4000-dn500/10-20"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.1-tn4000-dn500/10-20"


# train_data, dev_data, test_data = torch.load(os.path.join(path, "data.pt"))

# print(train_data.size(), dev_data.size(), test_data.size())

# c_train = Counter(train_data[:, 2].tolist())
# c_dev = Counter(dev_data[:, 2].tolist())
# c_test = Counter(test_data[:, 2].tolist())

# print(c_train)
# print(c_dev)
# print(c_test)

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/30-20-7/baseline/"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-eval_opt/30-20-7/opt_alpha_330_wm700"

# all_dev_IF = torch.load(os.path.join(path, "all_dev_IF.pt"), map_location="cpu")

# dev_IF = all_dev_IF[0]

# e = 1400

# data_path = "/home/lidong1/yuxian/sps-toy/processed_data/toy-add/tn4000-dn500-r1.3/30-20/data.pt"

# data = torch.load(data_path, map_location="cpu")


# train_labels = data[0][:, 2]

# # print(dev_IF[0].size())

# # exit(0)


# plt.hist(dev_IF[e], bins=4000)

# plt.plot(dev_IF[e], train_labels, "o")

# plt.savefig(os.path.join(path, f"dev_IF_bsl_{e}.png"))

# vocab = torch.load("/mnt/yuxian/data/tinystories/all_data/all_tokens.pt", map_location="cpu")

# print(len(vocab))

# g_opt = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_params_opt.pt", map_location="cpu")
# grad = torch.load("/home/lidong1/yuxian/sps-toy/toy/grad.pt", map_location="cpu")

# delta = g_opt - grad

# print(torch.sum(torch.abs(delta)))

all_grad_out_norm = []

import re
import numpy as np

# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-opt-0.0001-0/30-20-7/grad_log.txt"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/tiny-128-4k-ts-64/bs1000-lr0.2-tn4000-dn500/-opt-0.01-50/10-20-7/grad_log.txt"
# path = "/home/lidong1/yuxian/sps-toy/results/toy/trm/tiny-128-4k-ts-64/bs1000-lr0.05-tn4000-dn500/-opt-0.01-50/10-20-7/grad_log.txt"

# with open(path) as f:
#     for i, line in enumerate(f):
#         p = r".*grad out norm (\d+\.\d+) max.*"
#         m = re.match(p, line)
#         if m is not None:
#             all_grad_out_norm.append(float(m.group(1)))
    
#         if len(all_grad_out_norm) == 500:
#             break

# # all_grad_out_norm = all_grad_out_norm[200:]
# all_grad_out_norm = [np.log(x) for x in all_grad_out_norm]

# fit = np.polyfit(np.array(list(range(len(all_grad_out_norm)))), np.array(all_grad_out_norm), 1)

# print(fit)

# plt.plot(all_grad_out_norm)
# plt.savefig("grad_out_norm.png")

# theta_500 = torch.load("/home/lidong1/yuxian/sps-toy/toy/theta_500.pt", map_location="cpu")
# theta_1000 = torch.load("/home/lidong1/yuxian/sps-toy/toy/theta_1000.pt", map_location="cpu")

# print(torch.sum(torch.abs(theta_500 - theta_1000)))

# from tiny_story_model import ToyTokenizer

# tokenizer = ToyTokenizer("/mnt/yuxian/checkpoints/tiny_stories/tiny-128-4k",
#                          "/home/lidong1/yuxian/sps-toy/processed_data/toy-ts/mistral/small_64_4000_500_2/vocab.pt")

# data_train = torch.load("/home/lidong1/yuxian/sps-toy/processed_data/toy-ts/mistral/small_64_4000_500_2/dev.pt", map_location="cpu")

# print(data_train)

# print(tokenizer.decode(data_train[1].tolist()))

# toy_trm_silu = torch.load("/home/lidong1/yuxian/sps-toy/processed_data/toy-ts/model_init/toy-trm-silu.pt")
# tiny_128_4k = torch.load("/home/lidong1/yuxian/sps-toy/processed_data/toy-ts/model_init/tiny-128-4k.pt")

# print(toy_trm_silu.keys())
# print(tiny_128_4k.keys())

# new_tot_trm_silu = {
#     "base_model.w_q.weight": tiny_128_4k["base_model.model.layers.0.self_attn.q_proj.weight"],
#     "base_model.w_k.weight": tiny_128_4k["base_model.model.layers.0.self_attn.k_proj.weight"],
#     "base_model.w_v.weight": tiny_128_4k["base_model.model.layers.0.self_attn.v_proj.weight"],
#     "base_model.w_o.weight": tiny_128_4k["base_model.model.layers.0.self_attn.o_proj.weight"],
#     "base_model.mlp_gate.weight": tiny_128_4k["base_model.model.layers.0.mlp.gate_proj.weight"],
#     "base_model.mlp_1.weight": tiny_128_4k["base_model.model.layers.0.mlp.up_proj.weight"],
#     "base_model.mlp_2.weight": tiny_128_4k["base_model.model.layers.0.mlp.down_proj.weight"],
#     "base_model.word_embed.weight": tiny_128_4k["base_model.model.embed_tokens.weight"],
#     "base_model.lm_head.weight": tiny_128_4k["base_model.lm_head.weight"],
#     "base_model.input_layernorm.weight": tiny_128_4k["base_model.model.layers.0.input_layernorm.weight"],
#     "base_model.post_attention_layernorm.weight": tiny_128_4k["base_model.model.layers.0.post_attention_layernorm.weight"],
#     "base_model.output_norm.weight": tiny_128_4k["base_model.model.norm.weight"],
# }

# torch.save(new_tot_trm_silu, "/home/lidong1/yuxian/sps-toy/processed_data/toy-ts/model_init/toy-trm-silu-2.pt")

# g_vec_500 = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_vec_500_1000.pt", map_location="cpu")
# g_vec_1000 = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_vec_1000_1000.pt", map_location="cpu")

# g_vec_500 = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_vec_500.pt", map_location="cpu")
# g_vec_1000 = torch.load("/home/lidong1/yuxian/sps-toy/toy/g_vec_1000.pt", map_location="cpu")

# print(g_vec_500)
# print(g_vec_1000)

# print(torch.sum(torch.abs(g_vec_500 - g_vec_1000)).item())


init = torch.load("/home/aiscuser/sps/processed_data/toy-ts/model_init/toy-trm.pt", map_location="cpu")

wq = init["base_model.w_q.weight"]
embed = init["base_model.word_embed.weight"]

plt.hist(embed.flatten().tolist(), bins=1000)

plt.savefig("test_init.png")

print(torch.max(embed), torch.min(embed))

print(torch.std(embed).item())