import os
import time
import json
import torch

from tiny_story_trainer import ToyTSTrainer
from addition_trainer import ToyAdditionTrainer
from opt_alpha_trainer import OptAlphaTrainer
from eval_alpha_trainer import EvalAlphaTrainer
from arguments import get_args


from utils import print_args, initialize, save_rank

torch.backends.cudnn.enabled = False


def main():
    args = get_args()
    initialize(args, do_distributed=False)

    print(args.save)
    
    print_args(args)
    with open(os.path.join(args.save, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    args.time_stamp = cur_time
    
    if args.opt_alpha:
        trainer_cls = OptAlphaTrainer 
    elif args.eval_opt_alpha:
        trainer_cls = EvalAlphaTrainer
    else:
        if args.data_names == "addition":
            trainer_cls = ToyAdditionTrainer
        elif args.data_names == "tiny_story":
            trainer_cls = ToyTSTrainer
        else:
            raise NotImplementedError
    
    trainer = trainer_cls(args, device)
    trainer.train()
    
if __name__ == "__main__":
    main()