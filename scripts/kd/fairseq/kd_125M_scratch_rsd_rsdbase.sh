#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=${4-2}
HOSTFILE=${5-hostfile_8V100_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT \
                  --hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE"

# type
TYPE="kd_rsd"
# model
CKPT_NAME="fairseq/125M-nt"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# CKPT_NAME="355M-35k"
# CKPT="${BASE_PATH}/results/fairseq/kd_rsd/fairseq_355M-d/t500K-w10K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/fairseq_1.3B-fairseq_125M-kd0.5/35000"
# CKPT_NAME="355M-nt-30k"
# CKPT="${BASE_PATH}/results/fairseq/kd_rsd/fairseq_355M-d-nt/t500K-w10K-bs4-lr0.0003cosine3e-05-G2-N32-NN4-scr/fairseq_1.3B-fairseq_125M-kd0.5/30000"
TEACHER_CKPT_NAME="fairseq/1.3B"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
BASE_CKPT_NAME="rsd125M_5000"
BASE_CKPT="${BASE_PATH}/results/fairseq/kd_rsd/fairseq_125M-nt/t500K-w10K-bs2-lr0.0003cosine3e-05-G4-N16-NN2-scr/fairseq_1.3B-fairseq_125M-kd0.5/5000"
# data
DATA_DIR="${BASE_PATH}/processed_data/pretrain/owbt/chunked/fairseq-1025"
# hp
BATCH_SIZE=4
LR=0.0003
LR_MIN=0.00003
GRAD_ACC=4
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/fairseq/${TYPE}"
# seed
SEED=10


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type fairseq"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --base-model-path ${BASE_CKPT}"
OPTS+=" --base-ckpt-name ${BASE_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --from-scratch"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 10000"
OPTS+=" --bin-data"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 10000"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 2.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 500000"
OPTS+=" --kd-ratio 0.5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval 5000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num 1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --start-from-global-step 5000"
# OPTS+=" --no-eval-when-start"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
