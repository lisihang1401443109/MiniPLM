import os
from tqdm import tqdm

import torch
import torch.distributed as dist

from utils import get_rank, save_rank, print_rank, all_gather
from train_eval_utils.base_trainer import BaseTrainer
from data_utils.prompt_datasets import PromptDataset
from torch.utils.data import DataLoader, DistributedSampler


class SFTLMInferer(BaseTrainer):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device, do_train=False)
        self.set_datasets(do_train=False)
        self.setup_model_and_optimizer(set_optim=False)
        
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        self.eval_dataset = PromptDataset(args, self.tokenizer, args.data_split, args.data_dir, args.infer_num, ada_max_length=args.ada_max_length)

    def inference(self):
        eval_dataset = self.eval_dataset
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate)
        
        self.model.eval()
        all_losses = []
                    
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"LM Evaluation", disable=(not get_rank() == 0))):
                if i == 0 and self.dp_rank == 0:
                    self.first_print(model_batch, no_model_batch)
                eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
                if self.args.torch_compile is not None:
                    loss = loss.clone()
                all_losses.append(loss)
        
        all_losses = torch.cat(all_losses, dim=0)
        all_losses = all_gather(all_losses, dim=1, op="stack")
        avg_loss = all_losses.mean().item()

        if get_rank() == 0:
            res = {"avg_loss": avg_loss}
            eval_log_str = self.get_log(res, "infer")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            torch.save(all_losses, os.path.join(self.args.save, "scores.pt"))
            print_rank("Scores saved to {}".format(os.path.join(self.args.save, "scores.pt")))
        else:
            res = None
        
        dist.barrier()
        return res