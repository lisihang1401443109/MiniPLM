#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}

# type
TYPE="toy"
# hp
LR=0.05
BATCH_SIZE=1000
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
OPTS+=" --model-path ${BASE_PATH}/checkpoints/tiny_stories/tiny-128-4k"
OPTS+=" --ckpt-name toy-trm"
# OPTS+=" --ckpt-name tiny-128-4k"
# data
OPTS+=" --train-num 4000"
OPTS+=" --dev-num 500"
OPTS+=" --test-num 500"
OPTS+=" --data-names tiny_story"
OPTS+=" --data-dir ${BASE_PATH}/processed_data/toy-ts/mistral/small_64_4000_500_2"
OPTS+=" --load-toy-data 1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size 500"
OPTS+=" --grad-batch-size 100"
OPTS+=" --epochs 100"
OPTS+=" --log-interval 10"
OPTS+=" --outer-lr 0.01"
OPTS+=" --outer-epochs 40"
OPTS+=" --clip-grad -1"
OPTS+=" --max-length 64"
# OPTS+=" --warmup-iters 100"
# OPTS+=" --opt-alpha-wm-steps 50"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --opt-alpha"
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
