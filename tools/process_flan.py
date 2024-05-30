"""Processing data"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))
import time
from collections import defaultdict
import multiprocessing
import numpy as np
import seqio
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args
import argparse
import torch
from transformers import AutoTokenizer

from utils import print_args, copy_to_blob, naive_copy_to_blob
from data_utils.indexed_dataset import make_builder

import flan.v2.mixtures


##############################################################
##### Instantiate the submixtures with each template style
##############################################################

# ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
# ZS means a zero-shot prompt, FS means a few-shot prompt
# OPT means the answer options for tasks with multiple choice answers are included in the template
# NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.max_response_length = self.max_length - self.max_prompt_length

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        if self.args.model_type in ["llama", "mistral"]:
            Encoder.tokenizer.pad_token = Encoder.tokenizer.eos_token

    def encode(self, ex):
        task_name = ex["_task_name"].numpy()
        prompt = ex["inputs_pretokenized"]
        response = ex["targets_pretokenized"]
        prompt_tokens = Encoder.tokenizer.encode(
            prompt, add_special_tokens=False)
        response_tokens = Encoder.tokenizer.encode(
            response, add_special_tokens=False)

        if 1 + len(prompt_tokens) + len(response_tokens) + 1 <= self.max_length:
            tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [-1] + \
                response_tokens + [Encoder.tokenizer.eos_token_id]
        else:
            if len(response_tokens) + 1 <= self.max_response_length:
                prompt_tokens = prompt_tokens[-(self.max_length - len(response_tokens) - 2):]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [-1] + \
                    response_tokens + [Encoder.tokenizer.eos_token_id]
            elif len(prompt_tokens) + 1 <= self.max_prompt_length:
                response_tokens = response_tokens[:(self.max_length - len(prompt_tokens) - 2)]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [-1] + \
                    response_tokens + [Encoder.tokenizer.eos_token_id]
            else:
                prompt_tokens = prompt_tokens[-(self.max_prompt_length-1):]
                response_tokens = response_tokens[:self.max_response_length-1]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [-1] + \
                    response_tokens + [Encoder.tokenizer.eos_token_id]

        return task_name, tokens, len(prompt) + len(response)


def setup():
    seqio.MixtureRegistry.add(
        'cot_submix',
        tasks=[
            ('cot_zsopt', 1),    # mixing weight = 50%
            ('cot_fsopt', 1),    # mixing weight = 50%
        ])

    seqio.MixtureRegistry.add(
        'dialog_submix',
        tasks=[
            ('dialog_zsopt', 1),    # mixing weight = 50%
            ('dialog_fsopt', 1),    # mixing weight = 50%
        ])

    seqio.MixtureRegistry.add(
        'niv2_submix',
        tasks=[
            ('niv2_zsopt', 1),    # mixing weight = 50%
            ('niv2_fsopt', 1),    # mixing weight = 50%
        ])

    seqio.MixtureRegistry.add(
        'flan2021_submix',
        tasks=[
            ('flan_zsopt', 1),      # mixing weight = 25%
            ('flan_fsopt', 1),      # mixing weight = 25%
            ('flan_zsnoopt', 1),    # mixing weight = 25%
            ('flan_fsnoopt', 1),    # mixing weight = 25%
        ])

    seqio.MixtureRegistry.add(
        't0_submix',
        tasks=[
            ('t0_zsopt', 1),      # mixing weight = 25%
            ('t0_fsopt', 1),      # mixing weight = 25%
            ('t0_zsnoopt', 1),    # mixing weight = 25%
            ('t0_fsnoopt', 1),    # mixing weight = 25%
        ])

    # Define the Final Flan Collection Mixture
    seqio.MixtureRegistry.add(
        'flan2022_submix',
        tasks=[
            ('flan2021_submix', 0.4),  # mixing weight = 40%
            ('t0_submix', 0.32),       # mixing weight = 32%
            ('niv2_submix', 0.2),      # mixing weight = 20%
            ('cot_submix', 0.05),      # mixing weight = 5%
            ('dialog_submix', 0.03),   # mixing weight = 3%
        ])


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(s + "\n")


def main():
    setup()
    args = get_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["llama", "mistral"]:
        tokenizer.pad_token = tokenizer.eos_token

    print_args(args)

    selected_mixture = seqio.get_mixture_or_task('flan2021_submix')

    print("Building Dataset")
    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": 2056, "targets": 512},
        num_epochs=1,
        shuffle=True,
        # use_cached=True,
        copy_pretokenized=True,
        # The passthrough features let you track the source/task/template metadata for the example
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )
    print("Dataset Built")

    sid, ofid = 0, 0, 0
    log_bytes_processed, log_doc_proccessed = 0, 0
    tokens_count = 0
    
    bin_file = os.path.join(args.save, f"data_{ofid}.bin")
    idx_file = os.path.join(args.save, f"data_{ofid}.idx")
    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
    
    encoder = Encoder(args)
    pool = multiprocessing.Pool(
        args.data_process_workers, initializer=encoder.initializer)

    if args.max_sample_num is not None:
        data_iter = dataset.take(args.max_sample_num)
    else:
        data_iter = dataset
    
    encoded_docs = pool.imap_unordered(encoder.encode, enumerate(data_iter), 50)
    # To read out the data you can do something like this:
    source_counter = defaultdict(lambda: 0)

    global_start = time.time()
    for i, (task_name, tokens, bytes_processed) in enumerate(encoded_docs):
        sid += 1
        log_bytes_processed += bytes_processed
        log_doc_proccessed += 1
        
        source_counter[task_name] += 1
        builder.add_item(torch.IntTensor(tokens))
        tokens_count += len(tokens)

        if sid % args.chunk_num_per_shard == 0:
            print("Shard {} is done.".format(ofid))
            builder.finalize(idx_file)

            naive_copy_to_blob(args.base_path, bin_file, bin_file.replace(
                "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
            naive_copy_to_blob(args.base_path, idx_file, idx_file.replace(
                "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)

            ofid += 1

            if ofid >= args.max_shard_num:
                break

            bin_file = os.path.join(args.save, f"data_{ofid}.bin")
            idx_file = os.path.join(args.save, f"data_{ofid}.idx")
            builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

        if sid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = log_bytes_processed / elapsed / 1024 / 1024
            ds = log_doc_proccessed / elapsed

            s = f"Processed {sid} examples. {(tokens_count / 1e9):.4f}B tokens. " + \
                f"({ds:.2f} docs/s, {mbs:.4f} MB/s). Total Time: {current - global_start} s."

            print_and_save(s, args.save)

            log_bytes_processed, log_doc_proccessed = 0, 0
            proc_start = current
        
        if sid >= args.max_sample_num:
            break

    pool.terminate()
    pool.close()
    pool.join()
    pool = None

    # summarize
    print_and_save(f"Total time: {time.time() - global_start}.", args.save)
    print_and_save(f"Total processed paragraphs: {sid}.", args.save)
    print_and_save(f"Total tokens: {tokens_count / 1e9:.4f}B", args.save)

    print_and_save(f"Data Submixture Counts: {source_counter}")


if __name__ == "__main__":
    main()