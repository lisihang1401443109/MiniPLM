import multiprocessing
import os
import time
import torch
import json
import sys
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args
from copy import deepcopy


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path)

    def encode(self, line):
        line = line.strip()
        lines = line.split("\n")
        if len(lines[0]) == 0:
            return None, None, len(line)
        if len(lines) == 1:
            return None, None, len(line)

        role = lines[0][-1]
        if role in ["A", "B"]:
            if lines[1].strip() in ["[表情包]", "[图片]"]:
                return None, None, len(line)
            tokens = Encoder.tokenizer.encode("\n" + lines[1].strip() + "\n", add_special_tokens=False)
            tokens = tokens[2:]
        else:
            return None, None, len(line)
        return tokens, role, len(line)


def main():
    print("OK")
    args = get_args()
        
    args.save = os.path.join(args.save, args.model_type, str(args.max_length))

    os.makedirs(args.save, exist_ok=True)
    
    with open(os.path.join(args.data_dir, "data.txt")) as f:
        raw_data = f.read().split("\n\n")
    
    dev_num = 1000
    all_data = {
        "train":raw_data[:-dev_num],
        "dev": raw_data[-dev_num:],
    }
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    for split in all_data:
        inst_num = 0
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.save, f"{split}_{0}.bin")
        idx_file = os.path.join(args.save, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.save, f"{split}.jsonl"), "w")
        
        length = 0
        curr_role = None
        
        role_tokens = {
            "A": tokenizer.encode("A: ", add_special_tokens=False),
            "B": tokenizer.encode("B: ", add_special_tokens=False)
        }
        
        for lid, (tokens, role, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if tokens is None:
                continue
            
            if curr_role is None:
                curr_role = role
                history = [deepcopy(role_tokens[curr_role])]
            
            if length + len(tokens) + 1 > args.max_length:
                flatten_tokens = [t for h in history for t in h]
                json_file.write(json.dumps({"text": tokenizer.decode(flatten_tokens)}) + "\n")
                binary_builder.add_item(torch.IntTensor(flatten_tokens))
                history = [deepcopy(role_tokens[curr_role])]
                length = 0
                inst_num += 1

            if role == curr_role:
                history[-1].extend(tokens)
                length += len(tokens)
            else:
                history[-1].append(tokenizer.eos_token_id)
                history.append(deepcopy(role_tokens[role]) + tokens)
                length += len(tokens) + 1 + len(role_tokens[role])

            curr_role = role
            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()

if __name__ == '__main__':
    main()