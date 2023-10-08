import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import get_model
from dataclasses import dataclass


@dataclass
class MOSOutput(CausalLMOutputWithPast):
    full_probs: torch.FloatTensor = None


class MOSConfig(PretrainedConfig):
    def __init__(self, args, inner_model_path, num_experts, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.inner_model_path = inner_model_path
        self.num_experts = num_experts
        self.inner_model_config = AutoConfig.from_pretrained(self.inner_model_path)


class MOSModel(PreTrainedModel):
    def __init__(self, config: MOSConfig, device):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.inner_model = get_model(config.args, device, config.inner_model_path, config.inner_model_config)
        self.inner_model_type = self.config.args.model_type
        
        if isinstance(self.inner_model, GPT2LMHeadModel):
            self.hidden_size = config.inner_model_config.n_embd
        else:
            raise NotImplementedError
        print(self.hidden_size, self.num_experts)
        self.gate_mlp = nn.Linear(self.hidden_size, self.num_experts, dtype=torch.float16, device=device, bias=False)
        self.hidden_mlp = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float16, device=device, bias=False) for _ in range(self.num_experts)])
        self.act_func = nn.Tanh()
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        return self.inner_model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    
    def forward(self,
                input_ids,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                use_cache=None,
                **kwargs):
        if isinstance(self.inner_model, GPT2LMHeadModel):
            transformer_outputs = self.inner_model.transformer(
                input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache, **kwargs)
        else:
            raise NotImplementedError
        
        hidden_states = transformer_outputs[0] # [bsz, seq_len, hidden_size]
        
        gate_probs = F.softmax(self.gate_mlp(hidden_states), dim=-1, dtype=torch.float32) # [bsz, seq_len, num_experts]
        full_probs = 0
        for e in range(self.num_experts):
            expert_hidden_states = self.act_func(self.hidden_mlp[e](hidden_states)) # [bsz, seq_len, hidden_size]
            expert_lm_logits = self.inner_model.lm_head(expert_hidden_states) # [bsz, seq_len, vocab_size]
            expert_probs = F.softmax(expert_lm_logits, dim=-1, dtype=torch.float32) # [bsz, seq_len, vocab_size]
            full_probs += gate_probs[:, :, e].unsqueeze(-1) * expert_probs
        
        if kwargs.get("return_logits", True):
            lm_logits = torch.log(full_probs)
        else:
            lm_logits = None
        
        return MOSOutput(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            full_probs=full_probs,
        )