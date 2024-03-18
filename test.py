import torch
import numpy as np
import json
from transformers import AutoTokenizer

# m = torch.load("/home/lidong1/yuxian/sps/results/gpt2/train/mos_kd/gpt2-base-gpt2-xlarge-sft/e20-bs16-lr0.0005-G1-N2-kd0.5-mos10_noact/2142/pytorch_model.bin")


# print(m["hidden_mlp.2.weight"])
# print(m["hidden_mlp.3.weight"])

# arr = np.random.randint(0, 10, size=(10, 20))

# # save arr
# np.save("arr.npy", arr)

# # use memmap to load arr
# arr = np.load("arr.npy", mmap_mode="r")

# print(arr[0, 2])

# arr = np.memmap("/home/lidong1/yuxian/sps/results/fairseq/test/fairseq_125M/t500K-bs2-lr0.0003cosine3e-05-G1-N4-NN1-scr/data_order.npy", dtype=np.int32, mode="r")

# arr1 = np.load("/home/lidong1/yuxian/sps/results/fairseq/test/fairseq_125M/t500K-bs2-lr0.0003cosine3e-05-G1-N4-NN1-scr/data_order.npy")

# print(arr[0])
# print(arr1[0, 0])

# print(arr.shape)
# print(arr1.shape)

# mb0 = torch.load("model_batch_0.pt", map_location="cpu")
# rmb0 = torch.load("rsm_model_batch_0.pt", map_location="cpu")

# tokenizer = AutoTokenizer.from_pretrained("checkpoints/fairseq/125M/")

# print(tokenizer.pad_token_id)
# print(tokenizer.eos_token_id)


# print(mb0["input_ids"][0].size())

# print(tokenizer.decode(mb0["input_ids"][0]))

# print(rmb0["input_ids"])

# mb0 = torch.load("model_batch_3.pt", map_location="cpu")
# # nmb0 = torch.load("no_model_batch_0.pt", map_location="cpu")

# rmb0 = torch.load("rsm_model_batch_3.pt", map_location="cpu")

# tokenizer = AutoTokenizer.from_pretrained("checkpoints/fairseq/125M/")

# print(mb0["input_ids"][0].size())
# print(mb0["input_ids"][0][-20:])

# print(tokenizer.decode(mb0["input_ids"][0]))

# # print(nmb0["label"][0][-20:])
# # print(nmb0["loss_mask"][0][-20:])

# print(rmb0["input_ids"][0].size())
# print(rmb0["input_ids"][0][-20:])

# print(tokenizer.decode(rmb0["input_ids"][0]))

# torch.manual_seed(0)

# t = torch.randint(0, 100, size=(10, 20))
# state = torch.get_rng_state()

# t1 = torch.randint(0, 100, size=(10, 20))
# t2 = torch.randint(0, 100, size=(10, 20))
# t3 = torch.randint(0, 100, size=(10, 20))

# # torch.set_rng_state(state)

# t11 = torch.randint(0, 100, size=(10, 20))


# print(t1)
# print(t11)

# states = torch.load("/home/lidong1/yuxian/sps/results/fairseq/test/fairseq_125M/t500K-bs2-lr0.0003cosine3e-05-G1-N4-NN1-scr/8/rng_states_0.pt", map_location="cpu")
# torch.set_rng_state(states["torch"])

# t = torch.randint(0, 100, (1, 10))

# print(t)

# m = torch.load("/home/aiscuser/sps/model_batch_None.pt", map_location="cpu")

# print(m["input_ids"][0].tolist()[:20])

from data_utils.distributed_indexed import DistributedMMapIndexedDataset

tokenizer = AutoTokenizer.from_pretrained("checkpoints/mistral/7B/")

data = DistributedMMapIndexedDataset("/home/guyuxian/sps/processed_data/pretrain/owbt_corrupt/chunked/mistral-1127", f"data", 0, 1)

print(data[1])

print(tokenizer.decode(data[3]))

# with open("dd") as f:
#     lines = f.readlines()

# n = 0
# for line in lines:
#     x, s = line.split(" ")
#     if float(s) > 0.13:
#         n += 1

# print(n)

# tokenizer = AutoTokenizer.from_pretrained("checkpoints/fairseq/125M/")

# tokenizer = AutoTokenizer.from_pretrained("checkpoints/mistral/7B/")

# s = "I love you.\n I love you."

# tokens = tokenizer.encode(s, add_special_tokens=False)

# print(tokens)

# with open("/home/guyuxian/sps/tools/end_sent_token_fairseq.json") as f:
#     obj = json.load(f)

# tokens = tokenizer.convert_ids_to_tokens(obj)

# print(tokens)