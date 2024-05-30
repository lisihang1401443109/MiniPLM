import os
import sys
base_path = sys.argv[1]
sys.path.append(base_path)
from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from data_utils.indexed_dataset import make_builder
import random
import numpy as np
from tqdm import tqdm
from utils import naive_copy_to_blob

start = int(sys.argv[2])
end = int(sys.argv[3])
random.seed(981217 + start + end)

input_dir = os.path.join(base_path, "processed_data/pretrain/cc_head/chunked/mistral-1025")
output_dir = os.path.join(base_path, "processed_data/pretrain/cc_head/chunked-shuf/mistral-1025")

tmp_output_dir = os.path.join(base_path, "processed_data_1/pretrain/cc_head/chunked-shuf/mistral-1025")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_output_dir, exist_ok=True)


for i in tqdm(range(start, end)):
    data = DistributedMMapIndexedDataset(input_dir, f"data_{i}", do_probe=False, load_to_ram=True)
    data_list = [d for d in data]
    random.shuffle(data_list)
    bin_file = os.path.join(tmp_output_dir, f"data_{i}.bin")
    idx_file = os.path.join(tmp_output_dir, f"data_{i}.idx")
    builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
    builder.add_np_items(data_list)
    builder.finalize(idx_file)

    naive_copy_to_blob(base_path, bin_file, output_dir.replace(base_path, ""), rm_source=True)
    naive_copy_to_blob(base_path, idx_file, output_dir.replace(base_path, ""), rm_source=True)
    