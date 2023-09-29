#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME="base-init"
CKPT="${BASE_PATH}/minillm_ckpts/gpt2/train/minillm_init/gpt2-base/"
TEACHER_CKPT_NAME="xlarge-sft"
TEACHER_CKPT="${BASE_PATH}/minillm_ckpts/gpt2/train/sft/gpt2-xlarge/"
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/gpt2/"
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/train/minillm2/"
# hp
GRAD_ACC=1
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
CHUNK_SIZE=16


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --dev-num 100"
OPTS+=" --num-workers 0"
OPTS+=" --bin-data"
# hp
OPTS+=" --epochs 80000"
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --do-train"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --inner-epochs 4"
OPTS+=" --num-rollouts 16"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --decode-type sp"
OPTS+=" --lookahead 3"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_tvd.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
