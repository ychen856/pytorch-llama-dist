#!/bin/bash

#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 1 2>&1|tee u_pc_batch_10_2_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 2 2>&1|tee u_pc_batch_10_4_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 3 2>&1|tee u_pc_batch_30_2_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 4 2>&1|tee u_pc_batch_30_4_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 5 2>&1|tee u_jetson_batch_10_2_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 6 2>&1|tee u_jetson_batch_10_4_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 7 2>&1|tee u_jetson_batch_20_2_only.log
#python3 algrithm_ppl_test3.py --config config_lambda.yaml --selection 8 2>&1|tee u_jetson_batch_20_4_only.log

python3 lm_head_training.py --head 16 --config config_nrp.yaml
python3 lm_head_training.py --head 18 --config config_nrp.yaml
python3 lm_head_training.py --head 20 --config config_nrp.yaml
python3 lm_head_training.py --head 22 --config config_nrp.yaml
python3 lm_head_training.py --head 24 --config config_nrp.yaml
python3 lm_head_training.py --head 26 --config config_nrp.yaml
python3 lm_head_training.py --head 28 --config config_nrp.yaml
python3 lm_head_training.py --head 30 --config config_nrp.yaml