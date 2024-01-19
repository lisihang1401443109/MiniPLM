BASE_PATH=${1}

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_5k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e6000/-0.8_30-opt-0.4-0/10-20-7 \
#     --alpha-epochs "0.4,b,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
#     --epochs 6000

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_5k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.4-0/10-20-7 \
#     --alpha-epochs "0.4,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
#     --epochs 3000

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_5k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.2-0/10-20-7 \
#     --alpha-epochs "0.2,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
#     --epochs 3000

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_5k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-5k-ln-ts-64/bs512-lr0.1-tn16384-dn512-e3000/-0.8_30-opt-0.1-0/10-20-7 \
#     --alpha-epochs "0.1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
#     --epochs 3000

# bash /home/aiscuser/sps/toy/scripts/trm/ts/opt_4k_noise.sh /home/aiscuser/sps 2030 8 --epochs 1000 --outer-lr 0.4

# bash /home/aiscuser/sps/toy/scripts/trm/ts/opt_4.5k_noise.sh /home/aiscuser/sps 2030 8 --epochs 2000 --outer-lr 0.4

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_4k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-4k-ln-ts-64/bs512-lr0.1-tn4096-dn512-e1000/-0.8_30-opt-0.4-0/10-20-7 \
#     --alpha-epochs "0.4,5,10,15,19" \
#     --epochs 1000 \
#     --eval-no-IF

# bash /home/aiscuser/sps/toy/scripts/trm/ts/eval_4.5k_noise.sh /home/aiscuser/sps/ 2030 8 \
#     --load-alpha ${BASE_PATH}/results/toy/trm/toy-trm-4.5k-ln-ts-64/bs512-lr0.1-tn8192-dn512-e2000/-0.8_30-opt-0.4-0/10-20-7 \
#     --alpha-epochs "0.4,5,10,15,19" \
#     --epochs 2000 \
#     --eval-no-IF

bash scripts/trm/ts/opt_l2_5k_noise.sh /home/aiscuser/sps/ 2030 8 --outer-lr 0.4

bash ../scripts/pad.sh /home/aiscuser/sps/ 2030 8 0