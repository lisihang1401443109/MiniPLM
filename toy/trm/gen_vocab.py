from transformers import AutoTokenizer
import json
from tqdm import tqdm
import torch
from multiprocessing import Pool


class Encoder():
    def __init__(self, max_length, model_path):
        self.max_length = max_length
        self.model_path = model_path

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode(self, line):
        text = line["story"]
        tokens = Encoder.tokenizer.encode(text, add_special_tokens=False)
        if self.max_length > 0:
            tokens = tokens[:self.max_length]
        return tokens


all_tokens = set()

tokenizer_path = "/mnt/yuxian/checkpoints/mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

encoder = Encoder(-1, tokenizer_path)
pool = Pool(processes=20, initializer=encoder.initializer)
for idx in range(50):
    print("processing data{:0>2d}.json".format(idx))
    with open("/mnt/yuxian/data/tinystories/all_data/data{:0>2d}.json".format(idx)) as f:
        data = json.load(f)
    
    encoded_data = pool.map(encoder.encode, data, chunksize=50)
    for did, d in enumerate(encoded_data):
        if did % 10000 == 0:
            print("{}/{}".format(did, len(encoded_data)))
        all_tokens.update(d)
    print(len(all_tokens))
pool.close()

all_tokens = sorted(list(all_tokens))
all_tokens = [tokenizer.eos_token_id] + all_tokens
print("vocab size", len(all_tokens))
torch.save(all_tokens, "/mnt/yuxian/data/tinystories/all_data/all_tokens_mistral.pt")

new_vocab = tokenizer.convert_ids_to_tokens(all_tokens)

with open("/mnt/yuxian/data/tinystories/all_data/new_vocab_mistral.txt", "w") as f:
    f.write("\n".join(new_vocab))
