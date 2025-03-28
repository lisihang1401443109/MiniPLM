import datasets
import os
from tqdm import tqdm
import json

from pathlib import Path

datasets.config.DOWNLOADED_DATASETS_PATH = Path("/mnt/work/data/pile")
datasets.config.HF_DATASETS_CACHE = Path("/mnt/work/data/pile")


os.environ["HF_HOME"] = "/mnt/work/data/pile"
os.environ["HF_HUB_CACHE"] = "/mnt/work/data/pile/hub"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/work/data/pile/transformers"
os.environ["HF_DATASETS_CACHE"] = "/mnt/work/data/pile/datasets"


data = datasets.load_dataset("monology/pile-uncopyrighted", cache_dir="/mnt/work/data/pile")
print(data)

output_dir = "/mnt/work/data/pile"
os.makedirs(output_dir, exist_ok=True)

max_num_per_shard = 1000000
ofid = 0
did = 0

for split in ["train"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    for d in tqdm(data[split]):
        f.write(json.dumps(d) + "\n")
        did += 1
        if did >= max_num_per_shard:
            f.close()
            ofid += 1
            did = 0
            f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    
    f.close()
