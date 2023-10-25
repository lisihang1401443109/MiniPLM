import os
from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, save_rank
from torch.distributed import get_rank
from data_utils.prompt_datasets import PromptDataset


class PreTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
        if do_train and self.args.resume_training:
            self.resume_training()
    
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = PromptDataset(args, self.tokenizer, "data", args.data_dir, args.train_num, min_offset=10000)
            print_rank("train num", len(self.train_dataset))
            self.eval_dataset = PromptDataset(args, self.tokenizer, "data", args.data_dir, args.dev_num, max_offset=10000)
            print_rank("valid num", len(self.train_dataset))
        else:
            self.eval_dataset = PromptDataset(args, self.tokenizer, "data", args.data_dir, args.dev_num, max_offset=10000)
    
    def compute_loss(self, model_batch, no_model_batch):
        return self.compute_lm_loss(model_batch, no_model_batch), {}

    def evaluate(self):
        lm_res = self.evaluate_lm()
        if get_rank() == 0:
            res = {**lm_res}
            
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
