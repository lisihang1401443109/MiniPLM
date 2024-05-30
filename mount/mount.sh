MNT_DIR=$1

sudo mkdir -p $MNT_DIR
sudo blobfuse2 mount $MNT_DIR --config-file=config.yml --tmp-path=$MNT_DIR-tmp -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120