BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --type "tokenize" \
    --data-names "dolly" \
    --data-dir ${BASE_PATH}/downstream_data/dolly/ \
    --save ${BASE_PATH}/processed_data/dolly/ \
    --model-path ${BASE_PATH}/checkpoints/fairseq/125M/ \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type fairseq
