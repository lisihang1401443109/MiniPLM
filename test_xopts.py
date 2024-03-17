import torch
import torch.nn.functional as F
import xformers.ops as xops

query = torch.randn(4, 10, 2, 8).cuda() # bs, seq_len, num_heads, head_dim
key = torch.randn(4, 10, 2, 8).cuda() # bs, seq_len, num_heads, head_dim
value = torch.randn(4, 10, 2, 8).cuda() # bs, seq_len, num_heads, head_dim
p = 0.0

attn_bias = torch.tril(torch.ones(10, 10)).cuda() # seq_len, seq_len
attn_bias = attn_bias.masked_fill(attn_bias == 1, float('-inf'))
attn_bias = attn_bias[None, None, :, :]

def normal(q, k, v):
    scale = 1.0 / q.shape[-1] ** 0.5
    q = q * scale
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn = q @ k.transpose(-2, -1)
    # if attn_bias is not None:
    #     attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = attn @ v
    attn = attn.transpose(1, 2) # bs, seq_len, num_heads, head_dim
    return attn

def eff(q, k, v):
    y = xops.memory_efficient_attention(
        q, k, v,
        p=p
    )
    return y

attn_normal = normal(query, key, value)
attn_eff = eff(query, key, value)
print(attn_normal.size())
print(attn_normal[0, 0, 0, :])

print(attn_eff.size())
print(attn_eff[0, 0, 0, :])


# import torch
# import torch.nn as nn
# import xformers.ops

# # torch.use_deterministic_algorithms(True)

# torch.random.manual_seed(0)

# q = torch.rand(1, 2, 8, 4).cuda()
# k = torch.rand(1, 2, 8, 4).cuda()
# v = torch.rand(1, 2, 8, 4).cuda()

# scale = 1 / q.shape[-1] ** 0.5
# scores = torch.matmul(q * scale, k.transpose(-2, -1))
# attn_weights_v1 = nn.functional.softmax(scores, dim=-1)
# attn_weights_v1 = nn.functional.dropout(attn_weights_v1, p=0)
# attn_weights_v1 = torch.matmul(attn_weights_v1, v)

# attn_weights_v2 = xformers.ops.memory_efficient_attention(query=q, key=k, value=v)

# print(attn_weights_v1[0, 0, 0, :])
# print(attn_weights_v2[0, 0, 0, :])