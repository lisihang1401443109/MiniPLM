"""Processing data"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import multiprocessing
import time
import torch
import random
import numpy as np
import random
import json
from transformers import LlamaTokenizer, AutoTokenizer

import argparse
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args
from nltk import sent_tokenize

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from data_utils.indexed_dataset import make_builder
from utils import print_args, copy_to_blob


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.retriever_max_length = 256

    def initializer(self,):
        # Use Encoder class as a container for global data
        if self.args.model_type == "llama":
            Encoder.tokenizer = LlamaTokenizer.from_pretrained(self.args.model_path)
        else:
            Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        
        if self.args.model_type in ["llama"]:
            Encoder.tokenizer.pad_token = Encoder.tokenizer.eos_token

    def encode(self, id_with_line):
        doc_id, line = id_with_line
        
        doc = line.strip().replace("<@x(x!>", "\n")

        if len(doc.strip()) == 0:
            return None, doc_id, len(line)

        line = Encoder.tokenizer.encode(doc, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        
        return line, doc_id, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser = add_hp_args(add_model_args(add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args


def check_sent_end(tokenizer, tokens):
    s = tokenizer.decode(tokens)
    return len(tokens) == 1 or " " in s


def main():
    args = get_args()
    output_path = args.save
    if args.model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.model_type in ["llama"]:
        tokenizer.pad_token = tokenizer.eos_token

    rt_token_id = {
        "llama": 198,
        "fairseq": 198
    }[args.model_type]
    
    print_args(args)
    
    output_path = os.path.join(output_path, "chunked", args.model_type + "-" + str(args.max_length))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path.replace("processed_data_1", "processed_data"), exist_ok=True)
    
    with open(os.path.join(output_path.replace("processed_data_1", "processed_data"), "args_get_paragraph.json"), "w") as f:
        json.dump(vars(args), f)
    
    with open(os.path.join(args.base_path, "tools", f"end_sent_token_{args.model_type}.json"), "r") as f:
        end_sent_token = json.load(f)
    
    startup_start = time.time()

    print("Opening", args.data_dir)
    print("Output path:", output_path)
    
    sid, lid, ofid = 0, 0, 0
    log_bytes_processed, log_doc_proccessed = 0, 0
    padded_token_num = 0
    chunk_tokens_buffer = [tokenizer.bos_token_id]

    bin_file = os.path.join(output_path, f"data_{ofid}.bin")
    print(bin_file)
    idx_file = os.path.join(output_path, f"data_{ofid}.idx")
    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.data_process_workers, initializer=encoder.initializer)
    input_file = args.data_dir
    fin = open(input_file, 'r', encoding="utf-8")
    
    proc_start = time.time()
    global_start = time.time()
    
    print("Processing", input_file)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), 10)

    for doc_tokens, doc_id, bytes_processed in encoded_docs:
        lid += 1
        log_bytes_processed += bytes_processed
        log_doc_proccessed += 1
                                
        chunk_tokens_buffer.extend(doc_tokens)
        while len(chunk_tokens_buffer) >= args.max_length:
            new_chunk = chunk_tokens_buffer[:args.max_length]
            chunk_tokens_buffer = [tokenizer.bos_token_id] + chunk_tokens_buffer[args.max_length:]
            for i in range(len(new_chunk)-1, -1, -1):
                if (new_chunk[i] in [tokenizer.eos_token_id, rt_token_id]) or \
                    (new_chunk[i] in end_sent_token and check_sent_end(tokenizer, (new_chunk+chunk_tokens_buffer[1:2])[i:i+2])): # check if the end is fake
                    # 1. Who are you? I am the D.A. and he is //Bat Man. -> Who are you? // I am the D.A. and he is Bat Man.
                    # 2. Who are you? I am the D.//A. -> Who are you? // I am the D.A.
                    incomplete_sent = new_chunk[i+1:]
                    new_chunk = new_chunk[:i+1] + [tokenizer.pad_token_id] * (args.max_length - (i+1))
                    chunk_tokens_buffer = chunk_tokens_buffer[:1] + incomplete_sent + chunk_tokens_buffer[1:]
                    padded_token_num += args.max_length - (i+1)
                    
                    break
            
            assert new_chunk[0] == tokenizer.bos_token_id
            assert len(new_chunk) == args.max_length
                                
            sid += 1
            builder.add_item(torch.IntTensor(new_chunk))
            
            if sid % args.chunk_num_per_shard == 0:
                print("Shard {} is done.".format(ofid))
                builder.finalize(idx_file)
                
                copy_to_blob(args.base_path, bin_file, bin_file.replace("processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
                copy_to_blob(args.base_path, idx_file, idx_file.replace("processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
                
                ofid += 1
                bin_file = os.path.join(output_path, f"data_{ofid}.bin")
                idx_file = os.path.join(output_path, f"data_{ofid}.idx")
                builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
            
        if lid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = log_bytes_processed / elapsed / 1024 / 1024
            ds = log_doc_proccessed / elapsed
            
            s = f"Processed {lid} documents. {sid} chunks. " + \
                f"Padding fraction: {padded_token_num / (sid * args.max_length)}." + \
                f"({ds} docs/s, {mbs} MB/s). Total Time: {current - global_start} s."
        
            print(s, file=sys.stderr)
            
            log_bytes_processed, log_doc_proccessed = 0, 0
            proc_start = current
        

    print("Shard {} is done.".format(ofid))
    builder.finalize(idx_file)
    
    copy_to_blob(args.base_path, bin_file, bin_file.replace("processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
    copy_to_blob(args.base_path, idx_file, idx_file.replace("processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)

    fin.close()
    pool.close()
    pool.join()
    
    # summarize
    print("Total time:", time.time() - startup_start)
    print("Total processed paragraphs:", sid)
    print("Average paragraph length:", log_bytes_processed / sid)
        

if __name__ == '__main__':
    main()