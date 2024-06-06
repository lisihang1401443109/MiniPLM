#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
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

# type
TYPE="sft_lm_infer"
# model
CKPT_NAME="fairseq_1.3B_2856"
CKPT="${BASE_PATH}/results/sft/dolly/fairseq_1.3B/e10-bs4-lr5e-05cosine1e-07-G1-N8-NN1/2856"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/fairseq"
# hp
BATCH_SIZE=4
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --model-type fairseq"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-names dolly"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --infer-num -1"
OPTS+=" --bin-data"
OPTS+=" --split-token-id 65535"
OPTS+=" --trunc-data"
OPTS+=" --ada-max-length"
OPTS+=" --data-split train"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-infer"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type ${TYPE}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
