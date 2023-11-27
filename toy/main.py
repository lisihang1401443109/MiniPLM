import os
import time
import json
import torch

from linear.linear_model import LinearModel
from arguments import get_args


from utils import print_args, initialize, save_rank

torch.backends.cudnn.enabled = False


def main():
    args = get_args()
    initialize(args, do_distributed=False)

    print_args(args)
    with open(os.path.join(args.save, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))

    linear_model = LinearModel(args, device, dim=args.linear_dim)
    linear_model.set_theta_gd()
    
    print(args.train_num, args.test_num)
    train_x, train_y = linear_model.generate_data(args.train_num)
    test_x, test_y = linear_model.generate_data(args.test_num)
    
    linear_model.set_train_data(train_x, train_y)
    linear_model.set_test_data(test_x, test_y)
    linear_model.set_init_theta()

    linear_model.train()
    linear_model.test()
    
if __name__ == "__main__":
    main()