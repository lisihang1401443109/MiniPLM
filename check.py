from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from transformers import AutoTokenizer
import sys

tokenizer = AutoTokenizer.from_pretrained("checkpoints/fairseq/125M/")

ctx = DistributedMMapIndexedDataset("/home/aiscuser/sps/processed_data/pretrain/owbt/chunked/fairseq-2048", "data", False, 0)

idx = int(sys.argv[1])

print(ctx[idx])
print(len(ctx[idx]))
print(ctx[idx][-20:])
print(tokenizer.decode(ctx[idx]))
