import torch

d1 = torch.load("test.pt", map_location="cpu")

d2 = torch.load("/home/aiscuser/sps/processed_data/toy-ts/model_init/toy-trm-5k-ln.pt", map_location="cpu")


print(d1.keys())
print(d2.keys())

d1["base_model.blocks.0.attn.w_q.weight"] = d2["base_model.w_q.weight"]
d1["base_model.blocks.0.attn.w_k.weight"] = d2["base_model.w_k.weight"]
d1["base_model.blocks.0.attn.w_v.weight"] = d2["base_model.w_v.weight"]
d1["base_model.blocks.0.attn.w_o.weight"] = d2["base_model.w_o.weight"]

d1["base_model.blocks.0.ffn.mlp_1.weight"] = d2["base_model.mlp_1.weight"]
d1["base_model.blocks.0.ffn.mlp_2.weight"] = d2["base_model.mlp_2.weight"]

d1["base_model.blocks.0.ln_1.weight"] = d2["base_model.ln_1.weight"]
d1["base_model.blocks.0.ln_1.bias"] = d2["base_model.ln_1.bias"]

d1["base_model.blocks.0.ln_2.weight"] = d2["base_model.ln_2.weight"]
d1["base_model.blocks.0.ln_2.bias"] = d2["base_model.ln_2.bias"]

d1["base_model.ln_end.weight"] = d2["base_model.ln_end.weight"]
d1["base_model.ln_end.bias"] = d2["base_model.ln_end.bias"]

d1["base_model.word_embed.weight"] = d2["base_model.word_embed.weight"]
d1["base_model.pos_embed.weight"] = d2["base_model.pos_embed.weight"]
d1["base_model.lm_head.weight"] = d2["base_model.lm_head.weight"]

torch.save(d1, "/home/aiscuser/sps/processed_data/toy-ts/model_init/toy-trm-5k-ln-new.pt")