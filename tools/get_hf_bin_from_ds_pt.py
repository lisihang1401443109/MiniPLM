import os
import torch

model_dir = "/home/lidong1/yuxian/sps/results/fairseq/test/fairseq_125M/t500K-bs2-lr0.0003cosine3e-05-G1-N4-NN1-scr/8/"

ds_pt = torch.load(os.path.join(model_dir, "mp_rank_00_model_states.pt"), map_location="cpu")

hf_bin = ds_pt["module"]

torch.save(hf_bin, os.path.join(model_dir, "pytorch_model.bin"))