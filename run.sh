ls /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps/
nvidia-smi
# echo $PATH
# conda init
cat /root/.bashrc
source /root/.bashrc
export PATH="/root/miniconda3/bin:$PATH"
export TRITON_CACHE_DIR="/mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/.triton/autotune"
echo $PATH
ls /root/miniconda3/bin/
cd /mnt/chongqinggeminiceph1fs/geminicephfs/wxime-training/yuxiangu/sps
wandb disabled
bash scripts/pretrain/mistral/8.7M_scr_owbt.sh $PWD 2030 1