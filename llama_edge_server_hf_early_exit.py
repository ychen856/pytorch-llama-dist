# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import gc
import math
import threading
from datetime import datetime

import torch
import time
from pathlib import Path
import argparse
import random

from transformers.modeling_outputs import BaseModelOutputWithPast

from feature_pruning import *

#import http_receiver
import http_receiver2 as http_receiver
import http_sender_gateway
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

from multiprocessing import set_start_method
import sys
from natsort import natsorted
import os
from data import get_loaders

from eval_sep_hf import get_eval_data
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue

from prune_all import prune_wanda_allocation
from calculate_opt import Calcualte_opt, find_row
from early_exit import early_exit_cpu, early_exit_cuda, early_exit_lm_head
from timestamp_manager import Timestamp_manager
from threading import current_thread, Thread
from multiprocessing import current_process
parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

incoming_queue = Queue()
outgoing_queue_forward = Queue()
outgoing_queue_return = Queue()
calculate_opt = Calcualte_opt()
timestamp_manager = Timestamp_manager()
nsamples = 0
temp = []



def layer_reallocation(type, start_idx, end_idx_buff, max_layers, models):
    if type == 1: #add buffer layers
        #print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf,
            return_unused_kwargs=True
        )
        #print('config: ', config)

        checkpoint_list = []
        checkpoints = sorted(Path(args.ckpt_dir_hf_sep).glob("consolidated.*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir_hf_sep}"

        checkpoints = checkpoints[end_idx_buff + 1:]
        checkpoint_idx = end_idx_buff
        for checkpoint in checkpoints:
            ckpt_path = checkpoint

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            checkpoint_idx = checkpoint_idx + 1
            if checkpoint_idx >= max_layers:
                break
            if checkpoint_idx > end_idx_buff + 2:
                break

        start_idx = end_idx_buff + 1
        if end_idx_buff + 3 <= max_layers:
            end_idx_buff = end_idx_buff + 3
        else:
            end_idx_buff = max_layers


        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)


        for i in range(start_idx, end_idx_buff + 1):
            #print('i: ', i)
            try:
                if i == 0:
                    models.append(LlamaForCausalLM_emb(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[0].to(device)
                elif i == 33:
                    models.append((LlamaForCausalLM_norm(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[33].to(device)

                elif i == 34:
                    models.append((LlamaForCausalLM_linear(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[34].to(device)
                else:
                    models.append(LlamaForCausalLM_layer_0(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[i].to(device)
            except:
                end_idx_buff = i - 1
                break

    if type == 2: # drop layers
        #print('decrease buffer')
        models = models[:-1]
        end_idx_buff = end_idx_buff - 1
    if type == 3:   #reallocate model
        #print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf,
            return_unused_kwargs=True
        )
        #print('config: ', config)

        checkpoint_list = []
        checkpoints = sorted(Path(args.ckpt_dir_hf_sep).glob("consolidated.*.pth"))
        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir_hf_sep}"

        start_idx_buff = max(0, start_idx - 3)
        print('FFFFFFFFFFff: ', max_layers)
        checkpoints = checkpoints[start_idx_buff:max_layers]
        checkpoint_idx = start_idx_buff

        print('start idxzzzz: ', start_idx_buff)
        for layer in models:
            print('layer: ', layer)

        print('end idx buff: ', end_idx_buff)
        for checkpoint in checkpoints:
            print('checkpoint idx: ', checkpoint_idx)
            if checkpoint_idx > end_idx_buff:
                print('yaaay')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            elif models[checkpoint_idx] is None:
                print('nooon')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))

            checkpoint_idx = checkpoint_idx + 1

        end_idx_buff = max_layers


        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        models = models[:end_idx_buff]

        checkpoint_idx = 0
        for i in range(0, end_idx_buff + 1):
            if i < start_idx_buff:
                models[i] = None
                continue
            #print('i: ', i)
            load_layer = False
            try:
                if i == 0:
                    if i >= len(models):
                        models.append(LlamaForCausalLM_emb(config))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_emb(config)
                        load_layer = True

                    if load_layer is True:
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[0].to(device)
                        checkpoint_idx = checkpoint_idx + 1
                elif i == 33:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_norm(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_norm(config)
                        load_layer = True

                    if load_layer is True:
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[33].to(device)
                elif i == 34:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_linear(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_linear(config)
                        load_layer = True

                    if load_layer is True:
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[34].to(device)
                else:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_layer_0(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_layer_0(config)
                        load_layer = True

                    if load_layer is True:
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[i].to(device)
            except:
                end_idx_buff = i - 1
                break
        #prune_wanda_allocation(args, models, tokenizer, device=torch.device("cuda:0"))
    if type == 4:   #reload the whole model
        load_model(args.ckpt_dir_hf_sep, 0, end_idx_buff, torch.device("cuda:0"))

    '''for i in range(0, len(models)):
                model = models[i]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)'''

    return models, end_idx_buff


def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    #print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("consolidated.*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    checkpoint_idx = 0
    for checkpoint in checkpoints:
        ckpt_path = checkpoint
        #print(f'Loading checkpoint "{ckpt_path}"')

        checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
        checkpoint_idx = checkpoint_idx + 1
        if checkpoint_idx > end_idx:
            break


    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    models = []
    for i in range(0, start_idx):
        models.append(None)

    print('start idx: ', start_idx)
    for i in range(start_idx, end_idx + 1):
        print('i: ', i)
        #print('check point list [i]: ', checkpoint_list[i])
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[0].to(device)
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[33].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
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

def get_dataset(tokenizer):
    dataset = "wikitext2_hf"
    bs = 1
    seqlen = 1024

    _, testloader = get_loaders(
        dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer
    )
    # Get input IDs
    testenc = testloader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    nsamples = 5
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    input_list = []
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        print('input: ', inputs)
        inputs = inputs.reshape(j - i, seqlen)
        print('inputs: ', inputs)
        print('inputs: ', inputs.shape)
        input_list.append(inputs)


    return input_list


def get_lm_head_idx(end_idx):

    lm_heads = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    lm_head = 1
    lm_head_idx = 0

    for i in range(0, len(lm_heads)):
        if lm_heads[i] > end_idx:
            #lm_head = lm_heads[i - 1]
            #lm_head_idx = lm_head_idx - 1
            break
        elif lm_heads[i] == end_idx:
            lm_head = lm_heads[i]
            lm_head_idx = i
            break

        lm_head = lm_heads[i]
        lm_head_idx = i

    lm_head_idx = lm_head_idx + 1


    return lm_head, lm_head_idx
def load_lm_head(checkpoints_dir, end_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)
    print('??: ', end_idx)

    lm_head, lm_head_idx = get_lm_head_idx(end_idx)

    print('lm_head: ', lm_head)
    print('lm_head_idx: ', lm_head_idx)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("lm_head.*.pth"))
    checkpoints = natsorted(checkpoints)
    #checkpoints = checkpoints.sort(key=natural_keys)
    #checkpoints = sorted(Path(checkpoints_dir).glob("lm_head.*.pth"), key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"


    for i in range(0, len(checkpoints)):
        if i == 0 or i == lm_head_idx:
            ckpt_path = checkpoints[i]
            print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))



    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    lm_models = []

    for i in range(0, len(checkpoint_list)):
        if i == 0:
            lm_models.append((LlamaForCausalLM_norm(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)

        else:
            lm_models.append((LlamaForCausalLM_linear(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)

    return lm_head, lm_models


def get_server_statistic_from_q():
    while not http_sender_gateway.returning_queue.empty():
        [start_idx, server_comp_time, rtt] = http_sender_gateway.returning_queue.get()
        calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
        calculate_opt.comm_statistics = rtt - server_comp_time
        print('server_side: ', [start_idx, server_comp_time, rtt])

def task1_data_receiving(args):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T1 do nothing!')

    while 1:
        http_receiver.run(port=args.gateway_port)

def task1_data_sending(args):
    while 1:
        timeout_count = 0

        #print('zzz', calculate_opt.steady_state)
        #while outgoing_queue_forward.empty() and incoming_queue.qsize() > 0 and calculate_opt.steady_state:
        #while outgoing_queue.empty() and input_queue.qsize() > 0:
        while outgoing_queue_forward.qsize() < 3 and incoming_queue.qsize() > 0 and calculate_opt.steady_state:
        #while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0:
            timeout_count = timeout_count + 1

            start_time = time.time()
            #print('outgoing queue size: ', outgoing_queue.qsize())

            if incoming_queue.qsize() > 0: #and calculate_opt.incoming_count + 2 >= calculate_opt.outgoint_count:
                idx = incoming_queue.qsize()
                timestamp_manager.start_times = (idx, start_time)

                input = incoming_queue.get()
                #outgoing_queue_forward.put([0, incoming_queue.get(), None, None, idx, 0, 0])


                unpacked_data = decompress_and_deserialize(input)
                print('received data: ', unpacked_data)
                csr_out = unpacked_data[1]

                outgoing_queue_forward.put([0, csr_out, None, None, idx, 0, 0])

                end_time = time.time()
                #print('client computation time: ', end_time - start_time)
                # calculate_opt.client_comp_statistics = (-1, end_idx_buff, end_time - start_time)
                print('server idle!')
            else:
                break


        data = outgoing_queue_forward.get()
        #print('data: ', data)
        calculate_opt.outgoint_count = calculate_opt.outgoint_count + 1
        http_sender_gateway.send_data(args.server_ip, args.server_port, data, calculate_opt, timestamp_manager)

def task2_computation(models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, is_dummy=True):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T2 computaton...')
    cycle_count = 0
    input_count = 0
    layer_amount = 2
    start_idx_buff = start_idx
    opt_layer_amount = 2
    statistics_period = calculate_opt.statistic_period
    while(1):
        print('http sender outgoing queue size: ', outgoing_queue_forward.qsize())
        print('start time: ', time.time())
        start_time_0 = time.time()
        if is_dummy:
            while incoming_queue.empty():
                print('wait...')
                time.sleep(0.01)

            input = incoming_queue.get()
        else:
            input = http_receiver.get_in_queue_data()


        '''start_idx = input[0]
        csr_out = input[1]
        ids = input[2]
        mask = input[3]
        idx = input[4]
        is_early_exit = False
        is_oom = False'''

        unpacked_data = decompress_and_deserialize(input)
        print('received data: ', unpacked_data)

        start_idx = unpacked_data[0]
        csr_out = unpacked_data[1]
        ids = unpacked_data[2]
        mask = unpacked_data[3]
        idx = unpacked_data[4]
        is_early_exit = False
        is_oom = False

        if csr_out[2] is None:
            http_receiver.set_outgoing_queue([-1, None, None])
            max_layers = start_idx - 3 + max_layer_amount
            models, end_idx_buff = layer_reallocation(3, start_idx, end_idx_buff, max_layers, models)
            lm_head, _ = get_lm_head_idx(end_idx)
            if not lm_head == head_idx:
                head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")
            start_idx_buff = max(0, start_idx - 3)
            end_idx = start_idx + opt_layer_amount
            layer_amount = opt_layer_amount
            #http_receiver.set_outgoing_queue([-1, None, None])
            continue
        elif csr_out[0] is not None :
            #recover from csr/csc
            out = BaseModelOutputWithPast()
            out.last_hidden_state = csr_to_dense(csr_out).unsqueeze(0)
            #out.last_hidden_state = csc_to_dense(unpacked_out).unsqueeze(0)
            out.past_key_values = None
            out.hidden_states = None
            out.attentions = None
        else:
            out = csr_out


        print('start idx: ', start_idx)
        print('end idx: ', end_idx)
        #input = http_receiver.get_in_queue_data()
        #print('start compute time: ', time.time())
        start_time = time.time()

        # Forward pass through the model
        if start_idx == 0 or start_idx > max_layers or start_idx < start_idx_buff:
            print('direct sent!')
            #out, ids, mask = models[0](out)
            outgoing_queue_forward.put([start_idx, out, ids, mask, idx, 0, start_idx]) # forward the original input to the server
            continue

        start_comp_time = time.time()
        if start_idx > 0 and start_idx <= max_layers and start_idx >= start_idx_buff:
            #find opt
            # TODO

            end_time = time.time()
            #print('0: ', end_time - start_time)
            for k in range(start_idx, end_idx + 1):
                print('layer: ', k)
                try:
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                    if k == head_idx:
                        try:
                            is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, head_idx)
                            #print('is early: ', is_early_exit)
                        except Exception as e:
                            print('early oom!')
                            is_oom = True
                            is_early_exit = False

                            end_idx = k

                        if is_early_exit:
                            timestamp_manager.end_times = (idx, time.time())
                            break

                except Exception as e:
                    print('oom!!!')
                    is_oom = True

                    end_idx = k - 1

                    #print('updated end idx: ', end_idx)
                    break

            print('is early: ', is_early_exit)

        if not is_early_exit and end_idx >= 33:
            start_time = time.time()
            lm_logits = models[33](out.last_hidden_state)
            end_time = time.time()

            if end_idx >=34:
                start_time = time.time()
                lm_logits = models[34](lm_logits)
                end_time = time.time()

            #print('logits: ', lm_logits)
            #print('logit size: ', lm_logits.size())

        total_comp_time = time.time() - start_comp_time

        # print('out: ', out)
        print('end compute time: ', time.time())
        print('total computation time: ', total_comp_time)

        '''total_comp_time = time.time() - start_comp_time

        print('out: ', out)
        print('end compute time: ', time.time())
        print('total computation time: ', total_comp_time)'''


        if is_dummy:
            break


        if is_early_exit or end_idx >= 34:
            http_receiver.set_outgoing_queue([start_idx, total_comp_time, idx])


        if not is_early_exit and end_idx < 34 and start_idx != 0:
            outgoing_queue_forward.put([end_idx + 1, out, ids, mask, idx, total_comp_time, start_idx])
            #print('outgoing queue PUT!')
            #print('insert gateway statistics: ', [start_idx, end_idx, end_idx - start_idx, end_idx_buff, total_comp_time])
            calculate_opt.gateway_comp_statistics = (start_idx, end_idx, end_idx - start_idx, end_idx_buff, total_comp_time)

            #existed_statistic = find_row(calculate_opt.gateway_comp_statistics, 0, start_idx)
            existed_opt = find_row(calculate_opt.gateway_opt_table, 0, start_idx)
            if len(existed_opt) > 0:
                input_count = input_count + 1
                cycle_count = cycle_count + 1



            if is_oom:
                end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
                layer_amount = end_idx - start_idx
                is_oom = False

            if len(existed_opt) == 0:
                end_idx = start_idx + 2
            else:
                if (input_count + 1) % 2 == 0 and input_count < 20 and end_idx < max_layers and statistics_period <= 10:
                #if (input_count + 1) % 3 == 0 and input_count < 20 and end_idx < max_layers and statistics_period <= 20:
                    #print('testing higher value(i<30)')
                    calculate_opt.max_layer_amount = layer_amount
                    layer_amount = layer_amount + 1

                if cycle_count == (statistics_period - 8) and input_count > 20 and cycle_count % 2 == 0:
                #if cycle_count == (statistics_period - 12) and input_count > 20 and cycle_count % 3 == 0:
                    #print('testing lower value (i>30)')
                    layer_amount = max(1, layer_amount - 2)

                if cycle_count > (statistics_period - 8) and input_count >= 20 and end_idx < max_layers and cycle_count % 2 == 0:
                #if cycle_count > (statistics_period - 12) and input_count >= 20 and end_idx < max_layers and cycle_count % 3 == 0:
                    #print('testing higher value (i>30): ')
                    calculate_opt.max_layer_amount = layer_amount
                    layer_amount = layer_amount + 1

                end_idx = start_idx + layer_amount

        #if (input_count) % 10 == 0:
        if len(calculate_opt.server_comp_statistics) >= statistics_period:
            print('statistic')
            #statistics_period = statistics_period + 5
            end_idx, end_idx_buff, statistics_period = calculate_opt.calclate_opt_gateway(start_idx)
            opt_layer_amount = end_idx - start_idx
            layer_amount = opt_layer_amount
            end_idx_buff = min(max_layers, end_idx_buff)
            #while new_buff_idx < end_idx_buff:
            #    models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

            lm_head, _ = get_lm_head_idx(end_idx)
            if not lm_head == head_idx:
                head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")
            cycle_count = 0

        #max_layers = start_idx + max_layer_amount

        #if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
        if end_idx_buff < end_idx and end_idx_buff < max_layers:
            models, end_idx_buff = layer_reallocation(1, start_idx, end_idx_buff, max_layers, models)
        while end_idx_buff > end_idx + 3:  #remove buffer
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

        torch.cuda.empty_cache()


    calculate_opt.statistic_period = statistics_period



    #print('round time: ', time.time() - start_time_0)


'''def task2_computation(models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, head_idx, tokenizer, device, is_dummy=True):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T2 computaton...')
    cycle_count = 0
    input_count = 0
    layer_amount = end_idx - start_idx
    statistics_period = calculate_opt.statistic_period
    while(1):
        print('http sender outgoing queue size: ', outgoing_queue_forward.qsize())
        print('start time: ', time.time())
        start_time_0 = time.time()
        if is_dummy:
            while incoming_queue.empty():
                print('wait...')
                time.sleep(0.01)

            input = incoming_queue.get()
        else:
            input = http_receiver.get_in_queue_data()

        start_idx = input[0]
        out = input[1]
        ids = input[2]
        mask = input[3]
        idx = input[4]
        is_early_exit = False
        is_oom = False


        print('start idx: ', start_idx)
        print('end idx: ', end_idx)
        #input = http_receiver.get_in_queue_data()
        #print('start compute time: ', time.time())
        start_time = time.time()

        # Forward pass through the model
        if start_idx == 0 or start_idx > max_layers:
            #out, ids, mask = models[0](out)
            outgoing_queue_forward.put([start_idx, out, ids, mask, idx, 0]) # forward the original input to the server

        start_comp_time = time.time()
        if start_idx > 0 and start_idx <= max_layers:
            #find opt
            # TODO

            end_time = time.time()
            #print('0: ', end_time - start_time)
            for k in range(start_idx, end_idx + 1):
                print('layer: ', k)
                try:
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                    if k == head_idx:
                        try:
                            is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, head_idx)
                            print('is early: ', is_early_exit)
                        except Exception as e:
                            print('early oom!')
                            is_oom = True
                            is_early_exit = False

                            end_idx = k

                        if is_early_exit:
                            timestamp_manager.end_times = (idx, time.time())
                            break

                except Exception as e:
                    print('oom!!!')
                    is_oom = True

                    end_idx = k - 1

                    #print('updated end idx: ', end_idx)
                    break

        if not is_early_exit and end_idx >= 33:
            start_time = time.time()
            lm_logits = models[33](out.last_hidden_state)
            end_time = time.time()

            if end_idx >=34:
                start_time = time.time()
                lm_logits = models[34](lm_logits)
                end_time = time.time()

            #print('logits: ', lm_logits)
            #print('logit size: ', lm_logits.size())

        total_comp_time = time.time() - start_comp_time

        # print('out: ', out)
        print('end compute time: ', time.time())
        print('total computation time: ', total_comp_time)

        #total_comp_time = time.time() - start_comp_time

        #print('out: ', out)
        #print('end compute time: ', time.time())
        #print('total computation time: ', total_comp_time)


        if is_dummy:
            break


        if is_early_exit or end_idx >= 34:
            http_receiver.set_outgoing_queue([start_idx, total_comp_time, idx])


        if not is_early_exit and end_idx < 34 and start_idx != 0:
            cycle_count = cycle_count + 1
            input_count = input_count + 1

            outgoing_queue_forward.put([end_idx + 1, out, ids, mask, idx, total_comp_time])
            #print('outgoing queue PUT!')
            #print('insert gateway statistics: ', [start_idx, end_idx, end_idx - start_idx, end_idx_buff, total_comp_time])
            calculate_opt.gateway_comp_statistics = (start_idx, end_idx, end_idx - start_idx, end_idx_buff, total_comp_time)

            #existed_statistic = find_row(calculate_opt.gateway_comp_statistics, 0, start_idx)
            existed_opt = find_row(calculate_opt.gateway_opt_table, 0, start_idx)



            if is_oom:
                end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
                layer_amount = end_idx - start_idx
                is_oom = False

            if len(existed_opt) == 0:
                end_idx = start_idx + 2

            if (input_count + 1) % 2 == 0 and input_count < 20 and end_idx < max_layers and statistics_period <= 10:
                #print('testing higher value(i<30)')
                calculate_opt.max_layer_amount = layer_amount
                layer_amount = layer_amount + 1

            if cycle_count == (statistics_period - 8) and input_count > 20 and cycle_count % 2 == 0:
                #print('testing lower value (i>30)')
                layer_amount = max(1, layer_amount - 2)

            if cycle_count > (statistics_period - 8) and input_count >= 20 and end_idx < max_layers and cycle_count % 2 == 0:
                #print('testing higher value (i>30): ')
                calculate_opt.max_layer_amount = layer_amount
                layer_amount = layer_amount + 1

        end_idx = start_idx + layer_amount

        #if (input_count) % 10 == 0:
        if len(calculate_opt.server_comp_statistics) >= statistics_period:
            print('statistic')
            #statistics_period = statistics_period + 5
            end_idx, end_idx_buff, statistics_period = calculate_opt.calclate_opt_gateway(start_idx)
            end_idx_buff = min(max_layers, end_idx_buff)
            #while new_buff_idx < end_idx_buff:
            #    models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

            lm_head, _ = get_lm_head_idx(end_idx)
            if not lm_head == head_idx:
                head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")
            cycle_count = 0

        #max_layers = start_idx + max_layer_amount

        #if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
        if end_idx_buff < end_idx and end_idx_buff < max_layers:
            models, end_idx_buff = layer_reallocation(1, start_idx, end_idx_buff, max_layers, models)
        while end_idx_buff > end_idx + 3:  #remove buffer
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

        torch.cuda.empty_cache()


    calculate_opt.statistic_period = statistics_period



    #print('round time: ', time.time() - start_time_0)
'''

def task3_summerizing(models, test_loader, bs, device):
    while 1:
        while not http_sender_gateway.returning_queue.empty():
            [start_idx, server_comp_time, rtt] = http_sender_gateway.returning_queue.get()
            calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
            calculate_opt.comm_statistics = rtt - server_comp_time
            print('server_side: ',  [start_idx, server_comp_time, rtt])

if __name__ == '__main__':
    set_start_method('spawn')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)


    calculate_opt.statistic_period = 20
    end_idx_buff = args.end_idx_buff
    early_idx_buff = args.early_idx_buff
    max_layer_amount = args.max_layer_amount
    max_layers = args.max_layers
    start_idx = args.start_idx
    end_idx = args.end_idx
    head_idx = args.head_idx

    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, early_idx_buff, end_idx_buff, device)
    _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    print("loading success")
    # Create and start threads


    start_time = time.time()
    thread1 = threading.Thread(target=task1_data_receiving, args=[args])
    thread2 = threading.Thread(target=task1_data_sending, args=[args])
    #(models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, is_dummy=True)
    thread3 = threading.Thread(target=task2_computation, args=[models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, False])

    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    thread3.join()
    print('total_time: ', time.time() - start_time)
