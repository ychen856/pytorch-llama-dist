#!/bin/bash

python3.8 llama_edge_hf_early_exit_jetson2.py --config config_jetson2.yaml 2>&1|tee batch_20_1_all.log
python3.8 llama_edge_hf_early_exit_jetson2.py --config config_jetson2.yaml 2>&1|tee batch_20_2_all.log

python3.8 llama_edge_hf_early_exit_jetson2.py --config config_jetson2.yaml 2>&1|tee batch_20_3_all.log
python3.8 llama_edge_hf_early_exit_jetson2.py --config config_jetson2.yaml 2>&1|tee batch_20_4_all.log