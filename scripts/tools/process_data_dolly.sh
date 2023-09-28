BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --save ${BASE_PATH}/processed_data/dolly/ \
    --model-path ${BASE_PATH}/checkpoints/gpt2-large \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2
