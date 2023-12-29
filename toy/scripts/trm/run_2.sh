bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.1 --epochs 4000 --batch-size 256 --grad-batch-size 256
bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.2 --epochs 4000 --batch-size 256 --grad-batch-size 256
bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.4 --epochs 4000 --batch-size 256 --grad-batch-size 256


bash ../scripts/pad.sh /home/aiscuser/sps/ 2030 8 2