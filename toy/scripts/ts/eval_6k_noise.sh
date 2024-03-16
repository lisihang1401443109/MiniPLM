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
OPTS+=" --model-path ${BASE_PATH}/checkpoints/tiny_stories/tiny-128-6k"
OPTS+=" --ckpt-name toy-trm-6k-ln"
# data
OPTS+=" --train-num 32768"
OPTS+=" --dev-num 512"
OPTS+=" --test-num 512"
OPTS+=" --data-names toy-ts"
OPTS+=" --data-dir ${BASE_PATH}/processed_data/toy-ts/mistral/small_64_32768_512_2"
OPTS+=" --load-toy-data 1"
# OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.6-0/10-20-7"
# OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.1-0/10-20-7"
# OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.4-0/10-20-7"
# OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.2-0/10-20-7"
# OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.2-0/10-20-7"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size 64"
OPTS+=" --grad-batch-size 512"
OPTS+=" --epochs 8000"
OPTS+=" --log-interval 10"
OPTS+=" --clip-grad -1"
OPTS+=" --max-length 64"
OPTS+=" --avg-IF-calc-interval 2"
OPTS+=" --add-noise 0.8_30"
# OPTS+=" --num-samp-grad 16384"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --eval-opt-alpha"
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
