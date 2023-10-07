import torch
from transformers import GenerationConfig

from .storage import RolloutStorage, Element
from .rl_datasets import ModelBatch, RLPromptDataset
from utils import print_rank, save_rank
import os


class Sampler():
    def __init__(self, args, trainer, prompt_dataset: RLPromptDataset):
        self.args = args
        self.prompt_dataset = prompt_dataset
        self.prompt_dataloader = prompt_dataset.create_dataloader(
            batch_size=self.args.chunk_size, shuffle=True, drop_last=True, num_workers=self.args.num_workers)
        self.prompt_iterator = iter(self.prompt_dataloader)
        self.storage: RolloutStorage = trainer.storage
        self.model = trainer.model
        self.teacher_model = trainer.teacher_model
        self.tokenizer = trainer.tokenizer
        self.device = trainer.device
        self.epochs = 0
    
    def generate(self, prompt_batch, model_name="base"):
        
        model = self.model if model_name == "base" else self.teacher_model
        
        generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
            max_length=self.args.max_length,
            min_length=None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        
        gen_out = model.generate(**prompt_batch, generation_config=generation_config)
        
        return gen_out
    
    def run_sample(self, num_rollouts, iter_count):
        all_elems = []
        while len(all_elems) < num_rollouts:
            try:
                prompt_batch: ModelBatch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.prompt_dataloader)
                prompt_batch = next(self.prompt_iterator)
                self.epochs += 1
                print_rank(f"Sampler: Epoch {self.epochs}")
                save_rank(f"Sampler: Epoch {self.epochs}", os.path.join(self.args.save, "log.txt"))
            
            prompt_batch = self.prompt_dataset.move_to_device(prompt_batch, self.device)
            
            with torch.no_grad():
                gen_out = self.generate(prompt_batch.__dict__, model_name="teacher")
                full_ids = gen_out["sequences"]
                prompt_ids = prompt_batch.input_ids
                prompt_length = prompt_ids.shape[1]
                response_ids = full_ids[:, prompt_length:]
                response_length = response_ids.shape[1]
            
            num_elems = prompt_ids.shape[0]
            
            prompt_ids = prompt_ids.cpu()
            response_ids = response_ids.cpu()
            
            elems = [
                Element(
                    prompt_ids[i],
                    response_ids[i],
                ) for i in range(num_elems)
            ]
            
            all_elems.extend(elems)
            
        self.storage.push(all_elems)