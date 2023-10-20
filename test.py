import torch

m = torch.load("/home/lidong1/yuxian/sps/results/gpt2/train/mos_kd/gpt2-base-gpt2-xlarge-sft/e20-bs16-lr0.0005-G1-N2-kd0.5-mos10_noact/2142/pytorch_model.bin")


print(m["hidden_mlp.2.weight"])
print(m["hidden_mlp.3.weight"])