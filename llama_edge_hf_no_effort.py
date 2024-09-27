# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import math
import threading
from typing import Optional

import numpy as np
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import argparse

import http_receiver
from data import get_loaders
import torch.nn as nn
import safetensors
import http_sender
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig
import torch.nn.functional as F
import sys

from eval_sep_hf import get_eval_data
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue
from prune_all import prune_wanda_allocation
from calculate_opt import Calcualte_opt
from timestamp_manager import Timestamp_manager

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

input_queue = Queue()
outgoing_queue = Queue()
calculate_opt = Calcualte_opt()
timestamp_manager = Timestamp_manager()

def layer_reallocation(type, start_idx, end_idx_buff, models):
    if type == 1: #add buffer layers
        print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf,
            return_unused_kwargs=True
        )
        #print('config: ', config)

        checkpoint_list = []
        checkpoints = sorted(Path(args.ckpt_dir_hf_sep).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir_hf_sep}"

        checkpoints = checkpoints[end_idx_buff + 1:]
        checkpoint_idx = end_idx_buff
        for checkpoint in checkpoints:
            ckpt_path = checkpoint
            #print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            checkpoint_idx = checkpoint_idx + 1
            if checkpoint_idx > end_idx_buff + 2:
                break

        start_idx = end_idx_buff + 1
        end_idx_buff = end_idx_buff + 3

        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)


        for i in range(start_idx, end_idx_buff + 1):
            print('i: ', i)
            try:
                if i == 0:
                    models.append(LlamaForCausalLM_emb(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    # models[0].model.embed_tokens.weight = nn.Parameter(checkpoint_list[0]['model.embed_tokens.weight'])
                    models[0].to(device)
                elif i == 33:
                    models.append((LlamaForCausalLM_norm(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    # models[33].model.norm.weight = nn.Parameter(checkpoint_list[33]['model.norm.weight'])
                    models[33].to(device)

                elif i == 34:
                    models.append((LlamaForCausalLM_linear(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    # models[34].lm_head.weight = nn.Parameter(checkpoint_list[34]['lm_head.weight'])
                    models[34].to(device)
                else:
                    models.append(LlamaForCausalLM_layer_0(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)

                    models[i].to(device)
            except:
                end_idx_buff = i - 1
                break
        '''for i in range(0, len(models)):
            model = models[i]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)'''
    if type == 2: # drop layers
        print('decrease buffer')
        models = models[:-1]
        end_idx_buff = end_idx_buff - 1
    if type == 3:   #pruning
        prune_wanda_allocation(args, models, tokenizer, device=torch.device("cuda:0"))
    if type == 4:   #reload the whole model
        load_model(args.ckpt_dir_hf_sep, 0, end_idx_buff, torch.device("cuda:0"))


    return models, end_idx_buff



def get_llm2(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = 1024
    return model


def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    checkpoint_idx = 0
    for checkpoint in checkpoints:
        ckpt_path = checkpoint
        print(f'Loading checkpoint "{ckpt_path}"')

        checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
        checkpoint_idx = checkpoint_idx + 1
        if checkpoint_idx > end_idx:
            break

    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    models = []
    for i in range(start_idx, end_idx + 1):
        print('i: ', i)
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[0].model.embed_tokens.weight = nn.Parameter(checkpoint_list[0]['model.embed_tokens.weight'])
            models[0].to(device)
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[33].model.norm.weight = nn.Parameter(checkpoint_list[33]['model.norm.weight'])
            models[33].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[34].lm_head.weight = nn.Parameter(checkpoint_list[34]['lm_head.weight'])
            models[34].to(device)
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[i].to(device)


    '''for i in range(0, len(models)):
        model = models[i]
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)'''

    return models



def task1_data_sending(args):
    print('T1 start...')
    while 1:
        timeout_count = 0
        while outgoing_queue.empty():
            timeout_count = timeout_count + 1
            if timeout_count > 6000:
                return

            time.sleep(0.001)


        data = outgoing_queue.get()
        #print('data: ', data)
        http_sender.send_data(args.server_ip, args.server_port, data, calculate_opt, timestamp_manager)


def task2_computation(models, start_idx, end_idx, end_idx_buff, max_layers, device):
    is_oom = False
    trash_data = False
    batch_count = 30

    #while not input_queue.empty():
    while(1):
        if input_queue.qsize() == 0:
            # time.sleep(150)
            while len(timestamp_manager.end_times) < 20:
                time.sleep(0.0001)
            timestamp_manager.get_time_diff_every_n_inputs(20)
            timestamp_manager.clearAll()
            time.sleep(20)

            if batch_count <= 1:
                break

            test_loader = get_eval_data(tokenizer)
            bs = 1

            # loading inputs data
            seqlen = 1024
            # Get input IDs
            testenc = test_loader.input_ids

            # Calculate number of samples
            nsamples = testenc.numel() // seqlen
            nsamples = 20
            # List to store negative log likelihoods
            nlls = []
            print(f"nsamples {nsamples}")

            for i in range(0, nsamples, bs):
                if i % 50 == 0:
                    print(f"sample {i}")

                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                inputs = inputs.reshape(j - i, seqlen)

                input_queue.put(inputs)

            batch_count = batch_count - 1



        print('edge device:')
        start_time = time.time()
        idx = input_queue.qsize()
        timestamp_manager.start_times = (idx, start_time)
        input = input_queue.get()

        # Forward pass through the model
        out, ids, mask = models[0](input)


        end_time = time.time()
        print('client computation time: ', end_time - start_time)

        outgoing_queue.put([end_idx + 1, out, ids, mask, idx])

        data = outgoing_queue.get()
        # print('data: ', data)
        http_sender.send_data(args.server_ip, args.server_port, data, calculate_opt, timestamp_manager)



        torch.cuda.empty_cache()

    print('YAY')


def task3_summerizing(models, test_loader, bs, device):
    while 1:
        while not http_sender.returning_queue.empty():
            [start_idx, server_comp_time, rtt] = http_sender.returning_queue.get()
            calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
            calculate_opt.comm_statistics = rtt - server_comp_time
            print('server_side: ',  [start_idx, server_comp_time, rtt])



if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    max_layers = 3

    start_idx = 0
    end_idx_buff = 3
    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx_buff, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)


    print("loading success")
    test_loader = get_eval_data(tokenizer)
    bs = 1

    # loading inputs data
    seqlen = 1024
    # Get input IDs
    testenc = test_loader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    nsamples = 20
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        input_queue.put(inputs)

    start_idx = 0
    end_idx = 0
    # Create and start threads
    #thread1 = threading.Thread(target=task1_data_sending, args=[args])
    thread2 = threading.Thread(target=task2_computation, args=[models, start_idx, end_idx, end_idx_buff, max_layers, device])
    #thread3 = threading.Thread(target=task3_summerizing, args=[models, test_loader, bs, device])
    #thread1.start()
    thread2.start()
    #thread3.start()

    # Wait for both threads to finish (optional)
    #thread1.join()
    thread2.join()
    #thread3.join()

    print("Both tasks completed!")

    timestamp_manager.get_time_diff_every_n_inputs(10)

