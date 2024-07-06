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

import sys

from eval_sep_hf import get_eval_data
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue
from prune_all import prune_wanda_allocation
from calculate_opt import Calcualte_opt

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

outgoing_queue = Queue()
calculate_opt = Calcualte_opt()

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

        if device == "cuda":
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

    if device == "cuda":
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
    while 1:
        while outgoing_queue.empty():
            time.sleep(0.001)

        '''while not http_sender.returning_queue.empty():
            [start_idx, server_comp_time, rtt] = http_sender.returning_queue.get()
            calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
            calculate_opt.comm_statistics = rtt - server_comp_time'''

        data = outgoing_queue.get()
        http_sender.send_data(args.server_ip, args.server_port, data, calculate_opt)
def task2_computation(models, test_loader, bs, start_idx, end_idx, end_idx_buff, max_layers, device):
    is_oom = False
    trash_data = False

    seqlen = 1024
    # Get input IDs
    testenc = test_loader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    nsamples = 50
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    #prune_wanda_allocation(args, models, tokenizer, testenc[0], device=torch.device("cuda:0"))
    # Loop through each batch
    for i in range(0, nsamples, bs):
        print('========================================')
        print('end idx: ', end_idx)
        print('end idx buffer: ', end_idx_buff)



        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        start_time = time.time()
        if (start_idx == 0 and end_idx < 33):
            print('edge device:')
            start_time_sub = time.time()
            # Forward pass through the model
            try:
                out, ids, mask = models[0](inputs)
            except Exception as e:
                print(e)
                trash_data = True


            end_time_sub = time.time()
            #print('0: ', end_time_sub - start_time_sub)
            #for k in range(1, len(models)):
            for k in range(1, end_idx):
                start_time_sub = time.time()
                #print('start time: ', start_time_sub)
                try:
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                except Exception as e:
                    print('oom!!!')
                    is_oom = True

                    end_idx = k - 1

                    print('2: ', end_idx)
                    break

                end_time_sub = time.time()
                #print('end time: ', end_time_sub)
                #print(k, end_time_sub - start_time_sub)

        elif (start_idx != 0 and end_idx == 34):
            print('server device:')
            for k in range(start_idx, len(models) - 2):
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)

            lm_logits = models[33](out.last_hidden_state)
            lm_logits = models[34](lm_logits)
        else:
            print('single device:')
            # Forward pass through the model
            out, ids, mask = models[0](inputs)

            for k in range(1, len(models) - 2):
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)

            lm_logits = models[33](out.last_hidden_state)
            lm_logits = models[34](lm_logits)

        end_time = time.time()
        print('client computation time: ', end_time - start_time)

        if not trash_data:
            #calculate_opt.client_comp_statistics = (end_idx, end_idx_buff, end_time - start_time)
            outgoing_queue.put([end_idx + 1, out, ids, mask])
        trash_data = False

        if is_oom == False:
            end_idx = end_idx + 1


        if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
            models, end_idx_buff = layer_reallocation(1, start_idx, end_idx_buff, models)
        while end_idx_buff > end_idx + 3:  #remove buffer
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, models)
            print('end_idx_buff: ', end_idx_buff)
            print('end idx: ', end_idx)

        torch.cuda.empty_cache()

    print('YAY')



def task3_summerizing(models, test_loader, bs, device):
    while 1:
        while not http_sender.returning_queue.empty():
            [start_idx, server_comp_time, rtt] = http_sender.returning_queue.get()
            calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
            calculate_opt.comm_statistics = rtt - server_comp_time
            print('server_side: ',  [start_idx, server_comp_time, rtt])
    '''while 1:
        [start_idx, server_comp_time, rtt] = http_sender.get_queue_data()
        print('server start idx: ', start_idx)
        print('server comp: ', server_comp_time)
        print('rtt: ', rtt)'''

    '''seqlen = 1024
    # Get input IDs
    testenc = test_loader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    nsamples = 1
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        while 1:
            lm_logits = http_sender.get_queue_data()

            if len(lm_logits) > 0:
                #http_sender.pop_incoming_queue()
                break

        lm_logits = http_sender.get_queue_data()
        print('start summarizing...')

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        sys.stdout.flush()
        print('in for', i)

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    print('ppl: ', ppl.item())'''


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    max_layers = 18

    start_idx = 0
    end_idx_buff = 7
    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx_buff, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)


    print("loading success")
    test_loader = get_eval_data(tokenizer)
    bs = 1

    start_idx = 0
    end_idx = 5
    # Create and start threads
    thread1 = threading.Thread(target=task1_data_sending, args=[args])
    thread2 = threading.Thread(target=task2_computation, args=[models, test_loader, bs, start_idx, end_idx, end_idx_buff, max_layers, device])
    #thread3 = threading.Thread(target=task3_summerizing, args=[models, test_loader, bs, device])
    thread1.start()
    thread2.start()
    #thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    #thread3.join()

    print("Both tasks completed!")






    #ppl = eval_ppl_sep_hf(models, tokenizer, args, start_idx, end_idx, device)
    #print(f"ppl on wikitext {ppl}")


    '''models = get_llm2(args.ckpt_dir_sep, 0, 34, device, 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    ppl = eval_ppl_sep_hf(models, tokenizer, device)
    print(f"ppl on wikitext {ppl}")'''