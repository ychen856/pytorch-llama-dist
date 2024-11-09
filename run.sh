#!/bin/bash

python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 1 2>&1|tee u_pc_batch_10_2_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 2 2>&1|tee u_pc_batch_10_4_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 3 2>&1|tee u_pc_batch_30_2_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 4 2>&1|tee u_pc_batch_30_4_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 5 2>&1|tee u_jetson_batch_10_2_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 6 2>&1|tee u_jetson_batch_10_4_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 7 2>&1|tee u_jetson_batch_20_2_alg.log
python3 algorithm_ppl_test2.py --config config_nrp.yaml --selection 8 2>&1|tee u_jetson_batch_20_4_alg.log