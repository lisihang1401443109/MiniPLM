import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import ToyTransformer
from dataclasses import dataclass


@dataclass
class ToyOutput():
    logits: torch.FloatTensor


class ToyAddTransformer(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        if config["toy"]:
            base_config = config["base_model_config"]
            config = {
                "vocab_size": base_config.vocab_size,
                "max_len": base_config.max_position_embeddings,
                "hidden_size": base_config.hidden_size,
                "num_head": base_config.num_attention_heads,
            }
            if args.embed_proj:
                config.update({
                    "embed_size": 64,
                    "embed_proj": True,
                })
            else:
                config.update({
                    "embed_size": base_config.hidden_size,
                    "embed_proj": False,
                })
            self.base_model_config = "toy"
            self.base_model = ToyTransformer(config)
        else:
            self.base_model_config = config["base_model_config"]
            self.base_model = AutoModelForCausalLM.from_config(self.base_model_config)
        
    def forward(self, input_ids):
        if self.base_model_config == "toy":
            output = ToyOutput(logits=self.base_model(input_ids))
            return output
        else:
            return self.base_model(input_ids)
    
    def compute_loss(self, input_ids, labels, alpha=None):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = self.forward(input_ids).logits
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
        logits = functional_call(model, (params, buffers), xn).logits
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
        logits = functional_call(model, (params, buffers), xn.unsqueeze(0)).logits
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