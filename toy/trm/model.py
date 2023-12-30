import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call 
import torch.distributed as dist


class ToyTransformer(nn.Module):
    def __init__(self, config):
        super(ToyTransformer, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_head = config["num_head"]
        self.vocab_size = config["vocab_size"]
        self.head_dim = self.hidden_size // self.num_head
        self.max_len = config["max_len"]
        # causal mask as a buffer
        self.register_buffer("casual_mask", 
            torch.tril(torch.ones(1, 1, config["max_len"], config["max_len"], dtype=torch.long)), persistent=False)
        
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.activation = F.relu
        
        self.mlp_1 = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.mlp_2 = nn.Linear(4 * self.hidden_size, self.hidden_size, bias=False)
        
        self.embed_size = config["embed_size"]
        self.word_embed = nn.Embedding(config["vocab_size"], self.embed_size)
        self.pos_embed = nn.Embedding(config["max_len"], self.embed_size)
        self.embed_proj = config["embed_proj"]
        if config["embed_proj"]:
            self.embed_hidden_proj = nn.Linear(self.embed_size, self.hidden_size, bias=False)
            self.hidden_embed_proj = nn.Linear(self.hidden_size, self.embed_size, bias=False)
        self.lm_head = nn.Linear(self.embed_size, config["vocab_size"], bias=False)
        
        # self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0.0)
            else:
                pass
        
    def split_heads(self, x):
        x = x.view(x.size(0), x.size(1), self.num_head, self.head_dim)
        x = x.transpose(1, 2)
        return x # [batch_size, num_head, seq_len, head_dim]
    
    def merge_heads(self, x):
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), self.hidden_size)
        return x
    
    def forward(self, input_ids):
        input_embed = self.word_embed(input_ids)
        pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        pos_embed = self.pos_embed(pos_ids)
        hidden_states = input_embed + pos_embed
        
        if self.embed_proj:
            hidden_states = self.embed_hidden_proj(hidden_states)
        
        residual = hidden_states
        
        # Self-Attention
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn = torch.masked_fill(attn, self.casual_mask == 0, torch.finfo(torch.float32).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn, v) # [batch_size, num_head, seq_len, head_dim]
        
        attn_output = self.merge_heads(attn_output)
        
        hidden_states = self.w_o(attn_output)
        
        hidden_states = residual + hidden_states
        
        # Dense
        residual = hidden_states
        x = hidden_states
        x = self.mlp_1(x)
        x = self.activation(x)
        x = self.mlp_2(x)
        
        hidden_states = residual + x
        
        if self.embed_proj:
            hidden_states = self.hidden_embed_proj(hidden_states)
        
        output = self.lm_head(hidden_states)
        
        return output


    def compute_loss(self, input_ids, labels, alpha=None):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = self.forward(input_ids)
        logits = logits[:, -1, :]
        losses = loss_fn(logits, labels)
        if alpha is None:
            loss = torch.mean(losses)
        else:
            loss = torch.sum(alpha * losses)
        
        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum(preds == labels).item() / labels.size(0)
        
        return loss, acc, preds

    @staticmethod
    def compute_loss_func(params, buffers, model, xn, yn, alpha=None):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), xn)
        logits = logits[:, -1, :]
        losses = loss_fn(logits, yn)
        if alpha is None:
            loss = torch.mean(losses)
        else:
            loss = torch.sum(alpha * losses)
        return loss

    @staticmethod
    def compute_loss_func_single(params, buffers, model, xn, yn):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), xn.unsqueeze(0))
        logits = logits[:, -1, :]
        losses = loss_fn(logits, yn.unsqueeze(0))
        loss = torch.mean(losses)
        return loss

    def vector_to_params(self, vec):
        pointer = 0
        d = {}
        for n, p in self.named_parameters():
            d[n] = nn.Parameter(vec[pointer:pointer+p.numel()].view(p.size()))
            pointer += p.numel()
        
        return d
    
    def params_to_vector(self, params):
        vec = []
        for n, p in self.named_parameters():
            vec.append(params[n].view(-1))
        vec = torch.cat(vec, dim=0)
        return vec
