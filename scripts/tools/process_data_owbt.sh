python3 tools/process_data_owbt.py \
    --base-path $1 \
    --model-path checkpoints/fairseq/125M \
    --data-dir pretrain_data/openwebtext/raw.txt \
    --save processed_data_1/pretrain/owbt/ \
    --max-length 2048 \
    --log-interval 10000 \
    --data-process-workers 96 \
    --model-type fairseq \
    --chunk-num-per-shard 1000000