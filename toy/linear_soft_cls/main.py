import os
import time
import json
import torch
import wandb

from linear_cls_model import LinearCLSModel
from dyna_alpha import LinearCLSModelDynaAlpha
# from fix_alpha import LinearModelFixAlpha
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
    args.time_stamp = cur_time
    
    model_cls = {
        "linear_soft_cls": LinearCLSModel,
        "linear_soft_cls_da": LinearCLSModelDynaAlpha,
        # "linear_fa": LinearModelFixAlpha
    }[args.model_type]
    
    g_gd = torch.Generator(device=device)
    g_gd.manual_seed(args.seed_gd)
    g_data = torch.Generator(device=device)
    g_data.manual_seed(args.seed_data)
    # g_train_data = torch.Generator(device=device)
    # g_train_data.manual_seed(10)
    g_train_data = torch.Generator(device=device)
    g_train_data.manual_seed(args.seed_data)
    g_init = torch.Generator(device=device)
    g_init.manual_seed(args.seed)
    
    model = model_cls(args, device, dim=args.input_dim, real_dim=args.input_real_dim)
    model.set_theta_gd(g=g_gd)
    
    if args.load_toy_data is not None:
        data_path = os.path.join(args.base_path, f"processed_data/toy_data/{args.load_toy_data}/")
        train_x, train_y, dev_x, dev_y, test_x, test_y, theta_init = torch.load(os.path.join(data_path, "data.pt"))
    else:
        train_x, train_y = model.generate_data(
            args.train_num, args.train_noise, args.train_mu, args.train_sigma, g=g_data)
    
        dev_x, dev_y = model.generate_data(
            args.dev_num, args.dev_noise, args.dev_mu, args.dev_sigma, g=g_data)
        test_x, test_y = model.generate_data(
            args.dev_num, args.dev_noise, args.dev_mu, args.dev_sigma, g=g_data)
        theta_init = None

    print("train y 1s", (train_y == 1).sum())
    print("dev y 1s", (dev_y == 1).sum())
    print("test y 1s", (test_y == 1).sum())

    model.set_train_data(train_x, train_y)
    model.set_dev_data(dev_x, dev_y)
    # linear_model.set_dev_data(train_x, train_y)
    model.set_test_data(test_x, test_y)
    model.set_init_theta(theta_init, g=g_init)

    torch.save((train_x, train_y, dev_x, dev_y, test_x, test_y, model.theta_init), os.path.join(args.save, "data.pt"))
    data_save_path = os.path.join(
        args.base_path, "processed_data", "toy_data", f"{args.dev_mu}-{args.train_sigma}-{args.dev_sigma}-{args.train_num}-{args.seed}-{args.seed_data}-{args.seed_gd}")
    os.makedirs(data_save_path, exist_ok=True)
    print(train_x)
    torch.save((train_x, train_y, dev_x, dev_y, test_x, test_y, model.theta_init), os.path.join(data_save_path, "data.pt"))
    # exit(0)
    model.train()
    
if __name__ == "__main__":
    main()