for id in 1 2 3
do
rsync -avzP ./* node-${id}:~/sps/toy/ \
    --exclude "checkpoints" \
    --exclude "downstream_data" \
    --exclude "pretrain_data" \
    --exclude "processed_data" \
    --exclude "processed_data_1" \
    --exclude "results" \
    --exclude "*__pychache__/*" \
    --exclude "*.egg-info" \
    --exclude "*.pyc" \
    --exclude "azcopy" \
    --exclude "send.sh" \
    --exclude "wandb" \
    --omit-dir-times
done