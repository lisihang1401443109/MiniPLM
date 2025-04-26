#!/bin/bash

BASE_PATH=${1:-"/home/MiniPLM"}
MASTER_PORT=${2:-2030}
GPUS_PER_NODE=${3:-1}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

# Type
TYPE="pretrain"

# Model
CKPT_NAME="qwen/50M"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"

# Data
DATA_DIR="${BASE_PATH}/processed_data/ref_pretrain/pile/qwen-1025"
DATA_NAME="pile"
WANDB_NAME="50M-pretrain"

# Hyperparameters
BATCH_SIZE=8
LR=0.0006
LR_MIN=0.00006
GRAD_ACC=8

# Sequence length
MAX_LENGTH=1024

# Runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"

# Seed
SEED=10

# Options
OPTS=""
OPTS+=" --type ${TYPE}"
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --from-scratch"
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 2000"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 10000"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --do-train"
OPTS+=" --save-interval 5000"
OPTS+=" --log-interval 1000"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --wandb-group pretrain_scratch"
OPTS+=" --wandb-name ${WANDB_NAME}"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
