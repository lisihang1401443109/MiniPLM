#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}

# type
TYPE="toy"
# hp
LR=0.001
BATCH_SIZE=-1
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type linear_da"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --linear-dim 128"
# data
OPTS+=" --train-num 2048"
OPTS+=" --dev-num 256"
OPTS+=" --test-num 256"
OPTS+=" --train-mu 0.0"
OPTS+=" --train-sigma 1.0"
OPTS+=" --train-noise 0.1"
OPTS+=" --dev-mu 0.5"
OPTS+=" --dev-sigma 0.1"
OPTS+=" --dev-noise 0.01"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-alpha 0.0001"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --epochs 10000"
OPTS+=" --log-interval 1000"
# runtime
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="python3 ${BASE_PATH}/toy/main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
