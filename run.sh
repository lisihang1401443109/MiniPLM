ls /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps/
nvidia-smi
# echo $PATH
# conda init
# cat /root/.bashrc
source /root/.bashrc
export PATH="/root/miniconda3/bin:$PATH:/usr/sbin:/sbin"
export TRITON_CACHE_DIR="/mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/.triton/autotune"
echo $PATH
ls /root/miniconda3/bin/
cd /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps
git config --global --add safe.directory /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps
wandb disabled
pip3 install -e /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps/transformers


# bash scripts/pretrain/mistral/8.7M_scr_owbt.sh $PWD 2030 1
# bash scripts/sft/gpt2/base_dolly.sh $PWD 2030 4 \
#     --batch-size 8 \
#     --attn-impl "eager" \
#     --xops-attn \
#     --save-additional-suffix _test \
    # --torch-compile reduce-overhead \

# bash scripts/sft/gpt2/xlarge_dolly.sh $PWD 2030 4 \
#     --batch-size 8 \

bash scripts/sft/fairseq/1.3B_dolly.sh $PWD 2030 4 \
    --batch-size 8 \
    # --attn-impl "eager" \
    # --xops-attn \
    # --torch-compile reduce-overhead \