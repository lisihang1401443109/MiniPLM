#! /bin/bash

BASE_PATH=${1-"/home/MiniPLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=1
# HOSTFILE=${5-hostfile_8V100_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"
                  #--hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE

# type
TYPE="pretrain"
# model
CKPT_NAME="mamba/130M"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data/pretrain/pile/qwen-mamba-1025"
DATA_NAME="pile-qwen-mamba"
WANDB_NAME="mamba-130M-pretrain"
# hp
BATCH_SIZE=16
LR=0.002
LR_MIN=0.00001
GRAD_ACC=4
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type mamba"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --from-scratch"
# OPTS+=" --torch-compile reduce-overhead"
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 10000"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.95"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 100000"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --save-interval 5000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# OPTS+=" --fp32"
# wandb
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
