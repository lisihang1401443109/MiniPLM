import multiprocessing
import os
import time
import torch
import json
import sys
import random
import nltk
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        text = line["text"]
        text = nltk.sent_tokenize(text)
        prompt = text[0]
        response = " ".join(text[1:])
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = Encoder.tokenizer.encode(response, add_special_tokens=False)
        
        prompt_tokens = prompt_tokens[-256:]
        response_tokens = response_tokens[:256]
        
        prompt_str = Encoder.tokenizer.decode(prompt_tokens)
        response_str = Encoder.tokenizer.decode(response_tokens)
        
        return prompt_tokens, response_tokens, prompt_str, response_str, len(text)


def main():
    print("OK")
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt`
    args = get_args()

    os.makedirs(args.save, exist_ok=True)
                
    fin = open(os.path.join(args.data_dir, "c4-train.00000-of-01024.jsonl_0.jsonl"), "r")
    # encoder use the tokenizer to encode data
    encoder = Encoder(args)

    # 2. Mapping all datas with Encoder, with the help of multiprocessing
    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=50)
    proc_start = time.time()
    total_bytes_processed = 0
    
    bin_file = os.path.join(args.save, f"valid_{0}.bin")
    idx_file = os.path.join(args.save, f"valid_{0}.idx")

    binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

    # put tokenized data into binary_builder
    inst_num = 0
        
    json_file = open(os.path.join(args.save, f"valid.jsonl"), "w")
    
    for lid, (prompt_tokens, response_tokens, prompt_str, response_str, bytes_processed) in enumerate(encoded_docs):
        
        total_bytes_processed += bytes_processed
        
        full_tokens = prompt_tokens + [-1] + response_tokens
        binary_builder.add_item(torch.IntTensor(full_tokens))
        
        json_file.write(json.dumps({"prompt": prompt_str, "output": response_str}) + "\n")
        
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