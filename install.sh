export NCCL_DEBUG=""
pip3 install transformers
pip3 install torch==2.0.1
pip3 install deepspeed
pip3 install torchvision==0.15.2
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install sentencepiece
pip3 install protobuf==3.20.3
pip3 install peft

ln -s /mnt/yuxian/sps/results_residual/ results
ln -s /mnt/yuxian/sps/processed_data/ processed_data
ln -s /mnt/yuxian/data downstream_data
ln -s /mnt/yuxian/checkpoints checkpoints
ln -s /mnt/yuxian/pretrain_data pretrain_data
ln -s /mnt/yuxian/azcopy_linux_amd64_10.20.1/azcopy azcopy
