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
import json
from transformers import AutoTokenizer

from utils import print_args, copy_to_blob, naive_copy_to_blob
from data_utils.indexed_dataset import make_builder


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
        Encoder.split_token_id = len(Encoder.tokenizer)

        if self.args.model_type in ["llama", "mistral"]:
            Encoder.tokenizer.pad_token = Encoder.tokenizer.eos_token

    def encode(self, id_with_ex):
        id, ex = id_with_ex
        ex = json.loads(ex)
        # task_name = ex["_task_name"]
        task_name, prompt, response = ex
        prompt_tokens = Encoder.tokenizer.encode(
            prompt, add_special_tokens=False)
        response_tokens = Encoder.tokenizer.encode(
            response, add_special_tokens=False)

        if len(response_tokens) == 0:
            # print("*** WARNING: Empty response tokens. Skipping. ***")
            # print(task_name, Encoder.tokenizer.decode(prompt_tokens))
            # print("\n")
            return None, None, len(prompt)

        if 1 + len(prompt_tokens) + len(response_tokens) + 1 <= self.max_length:
            tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [Encoder.split_token_id] + \
                response_tokens + [Encoder.tokenizer.eos_token_id]
        else:
            if len(response_tokens) + 1 <= self.max_response_length:
                prompt_tokens = prompt_tokens[-(self.max_length - len(response_tokens) - 2):]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [Encoder.split_token_id] + \
                    response_tokens + [Encoder.tokenizer.eos_token_id]
            elif len(prompt_tokens) + 1 <= self.max_prompt_length:
                response_tokens = response_tokens[:(self.max_length - len(prompt_tokens) - 1)]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [Encoder.split_token_id] + \
                    response_tokens # no eos id
            else:
                prompt_tokens = prompt_tokens[-(self.max_prompt_length-1):]
                response_tokens = response_tokens[:self.max_response_length]
                tokens = [Encoder.tokenizer.bos_token_id] + prompt_tokens + [Encoder.split_token_id] + \
                    response_tokens
        
        assert len(response_tokens) >= 1
        assert len(tokens) <= self.max_length + 1, f"len(tokens)={len(tokens)} > self.max_length={self.max_length}"

        return task_name, tokens, len(prompt) + len(response)


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path.replace("processed_data_1", "processed_data"), "log.txt"), "a") as f:
        f.write(s + "\n")


def main():
    args = get_args()
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.save.replace("processed_data_1", "processed_data"), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["llama", "mistral"]:
        tokenizer.pad_token = tokenizer.eos_token

    print_args(args)

    fin = open(os.path.join(args.data_dir, "raw.jsonl"))

    sid, ofid = 0, 0
    log_bytes_processed, log_doc_proccessed = 0, 0
    tokens_count = 0
    
    bin_file = os.path.join(args.save, f"data_{ofid}.bin")
    idx_file = os.path.join(args.save, f"data_{ofid}.idx")
    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
    
    encoder = Encoder(args)
    pool = multiprocessing.Pool(
        args.data_process_workers, initializer=encoder.initializer)
    
    encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), 50)
    # To read out the data you can do something like this:
    source_counter = defaultdict(lambda: 0)

    _chunks = []
    global_start = time.time()
    proc_start = global_start
    for i, (task_name, tokens, bytes_processed) in enumerate(encoded_docs):
        log_bytes_processed += bytes_processed
        log_doc_proccessed += 1
        
        if tokens is None:
            continue
        
        sid += 1
        source_counter[task_name] += 1
        _chunks.append(np.array(tokens, dtype=np.uint16))
        tokens_count += len(tokens)

        if sid % args.chunk_num_per_shard == 0:
            builder.add_np_items(_chunks)
            print("Shard {} is done.".format(ofid))
            builder.finalize(idx_file)
            _chunks = []

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

    if sid % args.chunk_num_per_shard != 0:
        builder.add_np_items(_chunks)
        print("Shard {} is done.".format(ofid))
        builder.finalize(idx_file)
        _chunks = []

        naive_copy_to_blob(args.base_path, bin_file, bin_file.replace(
            "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
        naive_copy_to_blob(args.base_path, idx_file, idx_file.replace(
            "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)

    pool.terminate()
    pool.close()
    pool.join()
    pool = None
    fin.close()
    fin = None

    # summarize
    print_and_save(f"Total time: {time.time() - global_start}.", args.save)
    print_and_save(f"Total processed paragraphs: {sid}.", args.save)
    print_and_save(f"Total tokens: {tokens_count / 1e9:.4f}B", args.save)

    print_and_save(f"Data Submixture Counts: {source_counter}", args.save)


if __name__ == "__main__":
    main()