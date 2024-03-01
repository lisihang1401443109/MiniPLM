BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_zhubao.py \
    --type tokenize \
    --data-dir ${BASE_PATH}/data/zhubao/ \
    --save ${BASE_PATH}/processed_data/zhubao/ \
    --model-path ${BASE_PATH}/checkpoints/minicpm \
    --data-process-workers 32 \
    --max-length 256 \
    --dev-num 1000 \
    --model-type minicpm
