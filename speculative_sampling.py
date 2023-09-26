import torch
import torch.nn as nn
from utils import sample_from_draft_model, get_distribution, sample
from transformers import AutoTokenizer, GenerationConfig
from copy import deepcopy


def prepare_inputs(
        input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            previous_length = past_key_values[0][0].shape[-2]
            # print(input_ids.size(1))
            # print(previous_length)
            input_ids = input_ids[:, previous_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def extend_attention_mask(attention_mask, n):
    return torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], n))], dim=-1)


def get_prefix_attention_mask(attention_mask, n):
    return attention_mask[:, :n]


def get_prefix_key_values(past_key_values, n):
    return [tuple([kv[:, :, :n, :] for kv in kv_layer]) for kv_layer in past_key_values]


def update_model_kwargs(
    outputs,
    model_kwargs
):
    # update past_key_values
    model_kwargs["past_key_values"] = outputs.past_key_values

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = extend_attention_mask(attention_mask, 1)

    return model_kwargs


def draft_sample_loop(draft_model, input_ids, model_kwargs, eos_token_id_tensor, generation_config: GenerationConfig, lookahead):
    origin_length = input_ids.shape[-1]
    output_logits = []
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    
    while True:
        model_inputs = prepare_inputs(input_ids, **model_kwargs)
        outputs = draft_model(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        
        next_token_scores = next_token_logits.float() / (generation_config.temperature + 1e-6)
        probs = nn.functional.softmax(next_token_scores, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens * unfinished_sequences + generation_config.pad_token_id * (1 - unfinished_sequences)
        
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        output_logits.append(next_token_logits)
        model_kwargs = update_model_kwargs(
            outputs, model_kwargs
        )

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        # stop when each sentence is finished
        if (unfinished_sequences.max() == 0) or ((input_ids.shape[-1] - origin_length) >= lookahead):
            this_peer_finished = True
        
        if this_peer_finished:
            break
    
    output_logits = torch.stack(output_logits, dim=1)
    return {
        "sequences": input_ids,
        "logits": output_logits,
        "model_kwargs": model_kwargs,
    }


def speculative_sampling2(
    it,
    target_model,
    draft_model,
    input_ids,
    generation_config: GenerationConfig,
    tokenizer,
    lookahead=5,
    **kwargs):

    batch_size = len(input_ids)
    
    assert batch_size == 1, 'Batch size should be 1'

    draft_model_kwargs = generation_config.update(**kwargs)
    target_model_kwargs = deepcopy(draft_model_kwargs)
    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False  # used by synced_gpus only

    prev_target_logits = None
    acc_times, rej_times = 0, 0
    while True:
        input_len = input_ids.size(1)
        draft_outputs = draft_sample_loop(draft_model, input_ids, draft_model_kwargs, eos_token_id_tensor, generation_config, lookahead)
        draft_output_seq = draft_outputs['sequences']
        draft_logits = draft_outputs['logits']
        draft_model_kwargs = draft_outputs['model_kwargs']
        
        draft_token_len = draft_output_seq.size(1) - input_len
        
        target_model_kwargs["attention_mask"] = extend_attention_mask(target_model_kwargs["attention_mask"], draft_token_len)
        
        target_model_inputs = prepare_inputs(draft_output_seq, **target_model_kwargs)
        target_outputs = target_model(**target_model_inputs, return_dict=True)
        target_logits = target_outputs.logits[:, -draft_token_len-1:, :]
        target_model_kwargs["past_key_values"] = target_outputs.past_key_values
        
        if target_logits.shape[1] < draft_logits.shape[1] + 1:
            assert prev_target_logits is not None, ('Target model should have exactly lookahead + 1 tokens', target_logits.shape[1], draft_logits.shape[1] + 1)
            target_logits = torch.cat([prev_target_logits.unsqueeze(1), target_logits], dim=1)
            prev_target_logits = None
        assert target_logits.shape[1] == draft_logits.shape[1] + 1, ('Target model should have exactly lookahead + 1 tokens', target_logits.shape[1], draft_logits.shape[1] + 1)
        
        target_distribution = nn.functional.softmax(target_logits, dim=-1)
        draft_distribution = nn.functional.softmax(draft_logits, dim=-1)
        
        accepted_flag = 1
        
        for t in range(lookahead):
            numerator = target_distribution[:, t].gather(-1, draft_output_seq[:, input_len+t].unsqueeze(dim=-1)).squeeze(dim=-1)
            denominator = draft_distribution[:, t].gather(-1, draft_output_seq[:, input_len+t].unsqueeze(dim=-1)).squeeze(dim=-1)
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator) # [batch_size]
            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).all():
                next_tokens = draft_output_seq[:, input_len+t].unsqueeze(dim=-1)
                input_ids = torch.concat([input_ids, next_tokens], dim=-1)
                acc_times += 1
            ## Rejection
            else:
                new_dist = (target_distribution[:, t, :] - draft_distribution[:, t, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                next_tokens = torch.multinomial(new_dist, num_samples=1)
                input_ids = torch.concat([input_ids, next_tokens], dim=-1)
                accepted_flag = 0
                rej_times += 1
            
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # stop when each sentence is finished
            if (unfinished_sequences.max() == 0) or (len(input_ids[0]) >= generation_config.max_length):
                this_peer_finished = True
        
            if (not accepted_flag) or this_peer_finished:
                break
        

        draft_model_kwargs["attention_mask"] = get_prefix_attention_mask(draft_model_kwargs["attention_mask"], input_ids.size(1))
        target_model_kwargs["attention_mask"] = get_prefix_attention_mask(target_model_kwargs["attention_mask"], input_ids.size(1))
        
        seen_prefix_len = input_ids.size(1) if accepted_flag else input_ids.size(1) - 1
        draft_model_kwargs["past_key_values"] = get_prefix_key_values(draft_model_kwargs["past_key_values"], seen_prefix_len)
        target_model_kwargs["past_key_values"] = get_prefix_key_values(target_model_kwargs["past_key_values"], seen_prefix_len)

        if accepted_flag == 1 and not this_peer_finished:
            next_tokens = sample(target_logits[:, -1, :], temperature=generation_config.temperature)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.concat([input_ids, next_tokens], dim=-1)
            prev_target_logits = target_logits[:, -1, :]

            draft_model_kwargs["attention_mask"] = extend_attention_mask(draft_model_kwargs["attention_mask"], 1)
            target_model_kwargs["attention_mask"] = extend_attention_mask(target_model_kwargs["attention_mask"], 1)

        # print(f"Accepted continuations: {tokenizer.decode(input_ids[0,input_len:], skip_special_tokens=True)}")

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        # stop when each sentence is finished
        if (unfinished_sequences.max() == 0) or (len(input_ids[0]) >= generation_config.max_length):
            this_peer_finished = True

        if this_peer_finished:
            break
        
    #     if it == 1:
    #         print("*" * 10)
    
    # if it == 1:
    #     exit(0)
    
    return {
        "sequences": input_ids,
        "acc_times": acc_times,
        "rej_times": rej_times,
    }
        

def speculative_sampling(it, target_model, draft_model, initial_prompt_seq, max_tokens, tokenizer, lookahead=4, temperature=1.0, debug=True):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < max_tokens:
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, eos_token_id=tokenizer.eos_token_id, temperature=temperature)
                
        target_logits = target_model(draft_outputs).logits[:, N-1:, :]

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        
        for t in range(lookahead):
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                token = draft_outputs[:, N+t].unsqueeze(dim=-1)
                fin_prompt_seq = torch.concat([fin_prompt_seq, token], dim=-1)
                n += 1
                if token == tokenizer.eos_token_id:
                    accepted_flag = 0
                    break
            ## Rejection
            else:
                new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id], dim=-1)
                accepted_flag = 0
                break

        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        
        # print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

        n += 1
        
        if fin_prompt_seq[0, -1] == tokenizer.eos_token_id:
            break

    return {
        "sequences": fin_prompt_seq
    }