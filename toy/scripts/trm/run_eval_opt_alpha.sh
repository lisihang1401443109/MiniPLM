#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}

# type
TYPE="toy"
# hp
LR=0.05
BATCH_SIZE=-1
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
OPTS+=" --input-dim 128"
# data
OPTS+=" --train-num 4000"
OPTS+=" --dev-num 500"
OPTS+=" --test-num 500"
OPTS+=" --ratio-1-2 1.3"
OPTS+=" --load-toy-data 1"
OPTS+=" --load-alpha ${BASE_PATH}/results/toy/trm/d128/bs-1-lr0.05-tn4000-dn500/r1.3-opt/10-20-7"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --epochs 2000"
OPTS+=" --log-interval 100"
OPTS+=" --lam 0.0"
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
CMD="python3 ${BASE_PATH}/toy/trm/main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
