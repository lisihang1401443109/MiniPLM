#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2030}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# type
TYPE="toy"
# hp
LR=0.1
BATCH_SIZE=512
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10
SEED_DATA=20


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type trm"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --ckpt-name toy-trm-ln"
# data
OPTS+=" --train-num 16384"
OPTS+=" --dev-num 512"
OPTS+=" --test-num 512"
OPTS+=" --data-names addition"
OPTS+=" --data-dir ${BASE_PATH}/processed_data/toy-add-100/4096_512_1.3"
OPTS+=" --load-toy-data 1"
OPTS+=" --ratio-1-2 1.3"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size 64"
OPTS+=" --grad-batch-size 512"
OPTS+=" --epochs 3000"
OPTS+=" --log-interval 10"
OPTS+=" --clip-grad -1"
OPTS+=" --num-samp-grad 4096"
# runtime
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-data ${SEED_DATA}"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/toy/trm/main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
