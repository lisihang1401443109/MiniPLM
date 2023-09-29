#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME=${4-"gpt2-xl"}
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
DRAFT_CKPT_NAME="base-init"
DRAFT_CKPT="${BASE_PATH}/minillm_ckpts/gpt2/train/minillm_init/gpt2-base/"
MP_SIZE=1
# data
DATA_NAMES="c4"
DATA_DIR="${BASE_PATH}/processed_data/c4/"
# hp
EVAL_BATCH_SIZE=1
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/eval_sp/"
TYPE="eval_sp"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --model-type gpt2"
OPTS+=" --draft-model-path ${DRAFT_CKPT}"
OPTS+=" --draft-ckpt-name ${DRAFT_CKPT_NAME}"
OPTS+=" --draft-model-type gpt2"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 50"
OPTS+=" --data-process-workers -1"
OPTS+=" --bin-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
# OPTS+=" --eval-ppl"
OPTS+=" --eval-gen"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 20"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --decode-type sp"
OPTS+=" --temperature 1"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
