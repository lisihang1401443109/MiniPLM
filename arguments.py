# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
import numpy as np
from numerize.numerize import numerize


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default=None)
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-ckpt-name", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--base-model-path", type=str)
    group.add_argument("--base-ckpt-name", type=str)
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--draft-model-type", type=str, default=None)
    group.add_argument("--draft-ckpt-name", type=str, default=None)
    group.add_argument("--draft-model-path", type=str, default=None)
    group.add_argument("--mos-experts", type=int, default=None)
    
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-all', action="store_true")
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    group.add_argument("--from-scratch", action="store_true")
    
    group.add_argument("--resume-training", action="store_true")
    group.add_argument("--start-from-global-step", type=int, default=None)
    group.add_argument("--resume-dir", type=str, default=None)
    group.add_argument("--resume-tag", type=str, default=None)
    group.add_argument("--no-eval-when-start", action="store_true")
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--precompute-data-order", action="store_true")
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--test-num", type=int, default=-1)
    group.add_argument("--test-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--lm-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-tvd", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    group.add_argument("--only-prompt", action="store_true")
    
    group.add_argument("--chunk-num-per-shard", type=int, default=10000)
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=None,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument('--adam-beta', type=float, default=0.9),
    group.add_argument('--adam-beta2', type=float, default=0.999),
    group.add_argument('--adam-eps', type=float, default=1e-8),
    group.add_argument("--kd-ratio", type=float, default=None)
    group.add_argument("--kd-rsd-loss", type=float, default=None)

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")

    group.add_argument("--residual-base-weight", type=float, default=1.0)
    group.add_argument("--residual-num", type=int, default=1)
    group.add_argument("--teacher-temperature", type=float, default=1.0)
    group.add_argument("--base-temperature", type=float, default=1.0)
    group.add_argument("--rsd-mix-ratio", type=float, default=1.0)
    group.add_argument("--input-base-probs", action="store_true")

    return parser


def add_ppo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('ppo', 'ppo configurations')
    
    group.add_argument("--reward-scaling", type=float, default=None)
    group.add_argument("--cliprange-reward", type=float, default=1)
    group.add_argument("--inner-epochs", type=int, default=None)
    group.add_argument("--num-rollouts", type=int, default=None)
    group.add_argument("--cliprange", type=float, default=0.2)
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--gamma", type=float, default=0.95)
    
    return parser


def add_minillm_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('minillm', 'minillm configurations')
    
    group.add_argument("--length-norm", action="store_true")
    group.add_argument("--single-step-reg", action="store_true")
    group.add_argument("--teacher-mixed-alpha", type=float, default=None)
    group.add_argument("--lm-coef", type=float, default=1)
    
    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    group.add_argument("--decode-type", type=str, default="trm_ar")
    group.add_argument("--lookahead", type=int, default=1)
    
    return parser


def add_toy_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('toy', 'toy experiments')
    group.add_argument("--input-dim", type=int, default=512)
    group.add_argument("--input-real-dim", type=int, default=None)
    group.add_argument("--lr-alpha", type=float, default=0.001)
    group.add_argument("--lam", type=float, default=0.0)
    group.add_argument("--outer-epochs", type=int, default=5)
    group.add_argument("--linear-theta-scale", type=int, default=1)
    group.add_argument("--train-mu", type=float, default=0)
    group.add_argument("--train-sigma", type=float, default=1)
    group.add_argument("--train-noise", type=float, default=1)
    group.add_argument("--dev-mu", type=float, default=0)
    group.add_argument("--dev-sigma", type=float, default=1)
    group.add_argument("--dev-noise", type=float, default=0.1)
    group.add_argument("--ood", type=float, default=None)
    group.add_argument("--alpha-update-interval", type=int, default=1)
    
    group.add_argument("--dnn-hidden-dim", type=int, default=None)
    group.add_argument("--gd-dnn-hidden-dim", type=int, default=None)
    
    return parser


def base_save_path(args):
    return os.path.join(
        args.save,
        (f"{args.ckpt_name.replace('/', '_')}"),
        (f"e{args.epochs}" if args.epochs is not None else f"t{numerize(args.total_iters)}") + \
        (f"-w{numerize(args.warmup_iters)}" if args.warmup_iters > 0 else "") + \
        (f"-bs{args.batch_size}-lr{args.lr}{args.lr_decay_style}{args.lr_min}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
        (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
        (f"-scr" if args.from_scratch else "") + \
        args.save_additional_suffix
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_ppo_args(parser)
    parser = add_minillm_args(parser)
    parser = add_gen_args(parser)
    parser = add_toy_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    assert args.model_type is not None

    args.n_gpu = args.n_gpu * args.n_nodes
    
    assert args.model_type is not None
        
    if args.type == "eval_main":
        if args.ckpt_name is not None:
            tmp = args.ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)
        else:
            ckpt_name = None
        save_path = os.path.join(
            args.save,
            f"{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            ckpt_name,
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type == "eval_sp":
        if args.ckpt_name is not None:
            tmp = args.ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)
        else:
            ckpt_name = None
        
        if args.draft_ckpt_name is not None:
            tmp = args.draft_ckpt_name.split("/")
            if tmp[-1].isdigit():
                draft_ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                draft_ckpt_name = "_".join(tmp)
        
        save_path = os.path.join(
            args.save,
            f"{args.decode_type}-{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            f"{ckpt_name}" + (f"-{draft_ckpt_name}" if args.draft_ckpt_name is not None else ""),
            f"{args.lookahead}",
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type in ["sft", "pretrain"]:
        args.save = os.path.join(
            base_save_path(args),
            (f"-mos{args.mos_experts}" if args.mos_experts is not None else ""),
        )
    elif args.type in ["kd", "kd_pretrain", "kd_contrastive"]:
        args.save = os.path.join(
            base_save_path(args),
            f"{args.teacher_ckpt_name.replace('/', '_')}" + f"-kd{args.kd_ratio}",
            (f"-mos{args.mos_experts}" if args.mos_experts is not None else ""),
        )
    elif args.type in ["pt_rsd"]:
        args.save = os.path.join(
            base_save_path(args),
            f"rsd{args.residual_base_weight}-num{args.residual_num}",
        )
    elif args.type in ["kd_rsd"]:
        args.save = os.path.join(
            base_save_path(args),
            f"{args.teacher_ckpt_name.replace('/', '_')}" + f"-{args.base_ckpt_name.replace('/', '_')}" + f"-kd{args.kd_ratio}",
        )
    elif args.type == "gen":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}"),
            (f"t{args.temperature}-l{args.max_length}"),
        )
        args.save = save_path
    elif args.type == "minillm":
        ppo_prefix = f"pe{args.inner_epochs}" + \
                     (f"_nr{args.num_rollouts}" if args.num_rollouts is not None else "")
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}-{args.teacher_ckpt_name}"),
            (f"bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-lm{args.lm_coef}-len{args.max_length}" + \
                (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")),
            ppo_prefix + args.save_additional_suffix
        )
        args.save = save_path
        
        if args.warmup_iters > 0:
            assert args.scheduler_name is not None
    elif args.type == "toy":
        if args.model_type in ["linear", "linear_fa", "linear_da"]:
            model_info = f"d{args.input_dim}-{args.input_real_dim}-l{args.lam}"
            suffix = ""
            if args.model_type == "linear_fa":
                suffix += (f"oe{args.outer_epochs}-lra{args.lr_alpha}-tmu{args.train_mu}-tsig{args.train_sigma}-tnoi{args.train_noise}-dmu{args.dev_mu}-dsig{args.dev_sigma}-dnoi{args.dev_noise}")
            elif args.model_type == "linear_da":
                suffix += (f"lra{args.lr_alpha}-tmu{args.train_mu}-tsig{args.train_sigma}-tnoi{args.train_noise}-dmu{args.dev_mu}-dsig{args.dev_sigma}-dnoi{args.dev_noise}-aui{args.alpha_update_interval}")
        elif args.model_type in ["linear_cls", "linear_cls_da"]:
            model_info = f"d{args.input_dim}-{args.input_real_dim}-l{args.lam}"
            suffix = ""
            if args.model_type == "linear_cls_da":
                suffix += (f"lra{args.lr_alpha}-tmu{args.train_mu}-tsig{args.train_sigma}-dmu{args.dev_mu}-dsig{args.dev_sigma}-aui{args.alpha_update_interval}")
        elif args.model_type in ["linear_dnn", "linear_dnn_fa"]:
            model_info = f"d{args.input_dim}-l{args.lam}-h{args.dnn_hidden_dim}"
            suffix = ""
            if args.model_type == "linear_dnn_fa":
                suffix += (f"oe{args.outer_epochs}-lra{args.lr_alpha}-tmu{args.train_mu}-tsig{args.train_sigma}-tnoi{args.train_noise}-dmu{args.dev_mu}-dsig{args.dev_sigma}-dnoi{args.dev_noise}")
        elif args.model_type in ["dnn_dnn", "dnn_dnn_fa"]:
            model_info = f"d{args.input_dim}-gh{args.gd_dnn_hidden_dim}-l{args.lam}-h{args.dnn_hidden_dim}"
            suffix = ""
            if args.model_type == "dnn_dnn_fa":
                suffix += (f"oe{args.outer_epochs}-lra{args.lr_alpha}-tmu{args.train_mu}-tsig{args.train_sigma}-tnoi{args.train_noise}-dmu{args.dev_mu}-dsig{args.dev_sigma}-dnoi{args.dev_noise}")

        suffix += args.save_additional_suffix
        save_path = os.path.join(
            args.save,
            args.model_type,
            model_info,
            (f"bs{args.batch_size}-lr{args.lr}-tn{args.train_num}-dn{args.dev_num}"),
            suffix
        )
        args.save = save_path

    else:
        raise NotImplementedError

    return args
