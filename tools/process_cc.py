"""Processing data"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))
import json
import numpy as np
import random
import torch
import time
import multiprocessing
from utils import print_args, copy_to_blob, naive_copy_to_blob
from data_utils.indexed_dataset import make_builder
from nltk import sent_tokenize
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args
import argparse
from transformers import AutoTokenizer



random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.retriever_max_length = 256

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        if self.args.model_type in ["llama", "mistral"]:
            Encoder.tokenizer.pad_token = Encoder.tokenizer.eos_token

    def encode(self, id_with_json_line):
        doc_id, json_line = id_with_json_line
        line = json.loads(json_line)
        doc = line["text"]
        tokens = Encoder.tokenizer.encode(
            doc, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        # del doc
        # del line

        return tokens, doc_id, 10


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args


def check_sent_end(tokenizer, tokens):
    s = tokenizer.decode(tokens)
    return len(tokens) == 1 or " " in s


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(s + "\n")


def main():
    args = get_args()
    output_path = args.save
    random.seed(981217)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.model_type in ["llama", "mistral"]:
        tokenizer.pad_token = tokenizer.eos_token

    rt_token_id = {
        "llama": 13,
        "fairseq": 198,
        "mistral": 13,
    }[args.model_type]

    print_args(args)

    output_path = os.path.join(
        output_path, "chunked", args.model_type + "-" + str(args.max_length))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path.replace(
        "processed_data_1", "processed_data"), exist_ok=True)

    with open(os.path.join(output_path.replace("processed_data_1", "processed_data"), "args_get_paragraph.json"), "w") as f:
        json.dump(vars(args), f)

    with open(os.path.join(args.base_path, "tools", f"end_sent_token_{args.model_type}.json"), "r") as f:
        end_sent_token = json.load(f)

    startup_start = time.time()

    print("input path", args.data_dir)
    print("Output path:", output_path)

    sid, lid, ofid = 0, 0, 0
    log_bytes_processed, log_doc_proccessed = 0, 0
    padded_token_num = 0
    chunk_tokens_buffer = [tokenizer.bos_token_id]

    bin_file = os.path.join(output_path, f"data_{ofid}.bin")
    idx_file = os.path.join(output_path, f"data_{ofid}.idx")
    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

    encoder = Encoder(args)
    pool = multiprocessing.Pool(
        args.data_process_workers, initializer=encoder.initializer)
    
    global_start = time.time()
    files_names =[]
    for _, _, files in os.walk(args.data_dir):
        for file_name in files:
            files_names.append(file_name)
            
    random.shuffle(files_names)
    _chunks = []
    
    proc_start = global_start
    
    for fid, file_name in enumerate(files_names):
        print_and_save(f"Processing {file_name}. {fid}/{len(files_names)}", output_path)
        input_file = os.path.join(args.data_dir, file_name)
        fin = open(input_file)

        # use the tokenizer to encode the sentences
        encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), 50)

        for doc_tokens, doc_id, bytes_processed in encoded_docs:
            lid += 1
            log_bytes_processed += bytes_processed
            log_doc_proccessed += 1

            chunk_tokens_buffer.extend(doc_tokens)
            while len(chunk_tokens_buffer) >= args.max_length:
                new_chunk = chunk_tokens_buffer[:args.max_length]
                chunk_tokens_buffer = [tokenizer.bos_token_id] + \
                    chunk_tokens_buffer[args.max_length:]
                for i in range(len(new_chunk)-1, -1, -1):
                    if (new_chunk[i] in [tokenizer.eos_token_id, rt_token_id]) or \
                            (new_chunk[i] in end_sent_token and check_sent_end(tokenizer, (new_chunk+chunk_tokens_buffer[1:2])[i:i+2])):  # check if the end is fake
                        # 1. Who are you? I am the D.A. and he is //Bat Man. -> Who are you? // I am the D.A. and he is Bat Man.
                        # 2. Who are you? I am the D.//A. -> Who are you? // I am the D.A.
                        incomplete_sent = new_chunk[i+1:]
                        # new_chunk = new_chunk[:i+1] + [tokenizer.pad_token_id] * (args.max_length - (i+1))
                        new_chunk = new_chunk[:i+1]
                        chunk_tokens_buffer = chunk_tokens_buffer[:1] + incomplete_sent + chunk_tokens_buffer[1:]
                        padded_token_num += args.max_length - (i+1)

                        break

                assert new_chunk[0] == tokenizer.bos_token_id
                assert len(new_chunk) <= args.max_length

                sid += 1
                _chunks.append(np.array(new_chunk, dtype=np.uint16))

                if sid % args.chunk_num_per_shard == 0:
                    print_and_save("Shuffling chunks in shard {}.".format(ofid), output_path)
                    random.shuffle(_chunks)
                    print_and_save("Adding chunks to shard {}.".format(ofid), output_path)
                    builder.add_np_items(_chunks)
                    print_and_save("Shard {} is done.".format(ofid), output_path)
                    builder.finalize(idx_file)
                    _chunks = []

                    naive_copy_to_blob(args.base_path, bin_file, bin_file.replace(
                        "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
                    naive_copy_to_blob(args.base_path, idx_file, idx_file.replace(
                        "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)

                    ofid += 1
                    if ofid >= args.max_shard_num:
                        break

                    bin_file = os.path.join(output_path, f"data_{ofid}.bin")
                    idx_file = os.path.join(output_path, f"data_{ofid}.idx")
                    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

            if lid % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = log_bytes_processed / elapsed / 1024 / 1024
                ds = log_doc_proccessed / elapsed
                tokens = (sid * args.max_length - padded_token_num) / 1e9

                s = f"Processed {lid} documents. {sid} chunks. {tokens:.4f}B tokens. " + \
                    f"Padding fraction: {padded_token_num / (sid * args.max_length):.4f}." + \
                    f"({ds:.2f} docs/s, {mbs:.4f} MB/s). Total Time: {current - global_start} s."

                print_and_save(s, output_path)

                log_bytes_processed, log_doc_proccessed = 0, 0
                proc_start = current

            if ofid >= args.max_shard_num:
                break

        fin.close()
        fin = None
        
        if ofid >= args.max_shard_num:
            break

    if sid % args.chunk_num_per_shard != 0:
        print_and_save("Shuffling chunks in shard {}.".format(ofid), output_path)
        random.shuffle(_chunks)
        print_and_save("Adding chunks to shard {}.".format(ofid), output_path)
        builder.add_np_items(_chunks)
        print_and_save("Shard {} is done.".format(ofid), output_path)
        builder.finalize(idx_file)
        _chunks = []

        naive_copy_to_blob(args.base_path, bin_file, bin_file.replace(
            "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)
        naive_copy_to_blob(args.base_path, idx_file, idx_file.replace(
            "processed_data_1", "processed_data").replace(args.base_path, ""), rm_source=True)

    if fin is not None:
        fin.close()

    pool.terminate()
    pool.close()
    pool.join()
    pool = None

    # summarize
    print_and_save(f"Total time: {time.time() - startup_start}.", output_path)
    print_and_save(f"Total processed paragraphs: {sid}.", output_path)
    total_tokens = sid * args.max_length - padded_token_num
    print_and_save(f"Total tokens: {total_tokens / 1e9:.4f}B", output_path)
    print_and_save(f"Total padding fraction: {padded_token_num / (sid * args.max_length)}.", output_path)


if __name__ == '__main__':
    main()
