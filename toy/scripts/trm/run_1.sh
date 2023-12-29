bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.4
bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.6
bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.8
bash scripts/trm/run_opt_alpha_ts_dp.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.4 --epochs 3000 --batch-size 256 --grad-batch-size 256

bash ../scripts/pad.sh /home/aiscuser/sps/ 2030 8 1