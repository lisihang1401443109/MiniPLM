import torch
try:
    from transformers import mpu
except:
    print("WARNING: mpu not found")

from transformers import GenerationConfig
import torch.nn as nn
import torch.distributed as dist


def autoregressive_sampling(
    model,
    input_ids,
    generation_config: GenerationConfig,
    **kwargs,
):

    model_kwargs = generation_config.update(**kwargs)
    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False  # used by synced_gpus only
    
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)
        
        next_token_logits = outputs.logits[:, -1, :]
        
        if hasattr(model.config, "is_model_parallel") and model.config.is_model_parallel:
            gathered_next_token_logits = mpu.all_gather(
                next_token_logits.contiguous(),
                dim=-1,
                world_size=mpu.get_model_parallel_world_size(),
                group=mpu.get_model_parallel_group())
            # pre-process distribution
            next_token_scores = gathered_next_token_logits
            probs = nn.functional.softmax(next_token_scores.float(), dim=-1)
            partition_size = next_token_scores.size(-1) // mpu.get_model_parallel_world_size()
            next_token_scores = next_token_scores[:, mpu.get_model_parallel_rank()*partition_size:(mpu.get_model_parallel_rank()+1)*partition_size]
        else:
            next_token_scores = next_token_logits.float() / (generation_config.temperature + 1e-6)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
        
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if hasattr(model.config, "is_model_parallel") and model.config.is_model_parallel:
            dist.broadcast(next_tokens, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
    
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if len(input_ids[0]) >= generation_config.max_length:
            this_peer_finished = True
        
        if this_peer_finished:
            break
    
    return {
        "sequences": input_ids,
    }
    
    