MNT_DIR=$1

ln -s $MNT_DIR/sps/results/ results
ln -s $MNT_DIR/sps/processed_data/ processed_data
ln -s $MNT_DIR/data downstream_data
ln -s $MNT_DIR/checkpoints checkpoints
ln -s $MNT_DIR/pretrain_data pretrain_data