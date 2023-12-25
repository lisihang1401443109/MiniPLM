import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer


class ToyTokenizer():
    def __init__(self, base_path, vocab_path):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.vocab = torch.load(vocab_path, map_location="cpu")
        self.old2new_vocab_map = {v: k for k, v in enumerate(self.vocab)}
        self.new2old_vocab_map = {k: v for k, v in enumerate(self.vocab)}
        self.pad_token_id = 0
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        token_ids = self.base_tokenizer.encode(text, add_special_tokens=False)
        new_token_ids = [self.old2new_vocab_map[t] for t in token_ids]
        return new_token_ids

    def decode(self, token_ids):
        old_token_ids = [self.new2old_vocab_map[t] for t in token_ids]
        text = self.base_tokenizer.decode(old_token_ids)
        return text


class ToyTSTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model_config = config["base_model_config"]
        self.base_model = AutoModelForCausalLM.from_config(self.base_model_config)
        
    def forward(self, input_ids):
        return self.base_model(input_ids)
    
    def compute_loss(self, input_ids, labels, alpha=None):
        loss_mask = (labels != -100).float()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = self.forward(input_ids).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(labels.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        if alpha is None:
            loss = torch.mean(losses)
        else:
            loss = torch.sum(alpha * losses)

        return loss
    
    @staticmethod
    def compute_loss_func(params, buffers, model, input_ids, labels, alpha=None):
        loss_mask = (labels != -100).float()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), xn).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), yn.view(-1))
        losses = losses.view(yn.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        if alpha is None:
            loss = torch.mean(losses)
        else:
            loss = torch.sum(alpha * losses)
        return loss
    
    @staticmethod
    def compute_loss_func_single(params, buffers, model, input_ids, labels):
        loss_mask = (labels != -100).float()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), xn.unsqueeze(0)).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), yn.view(-1))
        losses = losses.view(yn.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
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
