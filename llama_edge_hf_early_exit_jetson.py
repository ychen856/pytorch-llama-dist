# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import math
import threading
import torch
import time
from pathlib import Path
import argparse

import http_sender
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

import sys

from eval_sep_hf import get_eval_data
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue
from prune_all import prune_wanda_allocation
from calculate_opt import Calcualte_opt
from early_exit import early_exit_cpu, early_exit_cuda, early_exit_lm_head


from memory_profiler import profile
import gc
parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

input_queue = Queue()
outgoing_queue = Queue()
calculate_opt = Calcualte_opt()


def layer_reallocation(type, start_idx, end_idx_buff, max_layers, models):
    if type == 1: #add buffer layers
        print('increase buffer')
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
            print('i: ', i)
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
        print('decrease buffer')
        models = models[:-1]
        end_idx_buff = end_idx_buff - 1
    if type == 3:   #pruning
        prune_wanda_allocation(args, models, tokenizer, device=torch.device("cuda:0"))
    if type == 4:   #reload the whole model
        load_model(args.ckpt_dir_hf_sep, 0, end_idx_buff, torch.device("cuda:0"))

    '''for i in range(0, len(models)):
                model = models[i]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)'''
    gc.collect()
    return models, end_idx_buff


def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("consolidated.*.pth"))
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


def load_lm_head(checkpoints_dir, head_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("lm_head.*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    for checkpoint in checkpoints:
        ckpt_path = checkpoint
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
            break

    return lm_models


def get_server_statistic_from_q():
    while not http_sender.returning_queue.empty():
        [start_idx, server_comp_time, rtt] = http_sender.returning_queue.get()
        calculate_opt.server_comp_statistics = (start_idx, server_comp_time)
        calculate_opt.comm_statistics = rtt - server_comp_time
        print('server_side: ', [start_idx, server_comp_time, rtt])

def task1_data_sending(args):
    while 1:
        timeout_count = 0
        while outgoing_queue.empty():
            timeout_count = timeout_count + 1
            '''if not input_queue.empty() and calculate_opt.incoming_count >= calculate_opt.outgoint_count:  # if server idle

                outgoing_queue.put([0, input_queue.get(), None, None])
                print('server idle 0w0!')'''

            if timeout_count > 12000:
                print('task 1 end...')
                return

            time.sleep(0.001)

        data = outgoing_queue.get()
        calculate_opt.outgoint_count = calculate_opt.incoming_count + 1
        http_sender.send_data(args.server_ip, args.server_port, data, calculate_opt)


def task2_computation(models, lm_models, start_idx, end_idx, end_idx_buff, head_idx, max_layers, device):

    is_oom = False
    #prune_wanda_allocation(args, models, tokenizer, testenc[0], device=torch.device("cuda:0"))
    # Loop through each batch
    cycle_count = 0
    input_count = 0
    count = 0
    while not input_queue.empty():
        is_early_exit = False
        count = count + 1
        print('========================================')
        print('input count: ', count)
        print('end idx: ', end_idx)
        print('end idx buffer: ', end_idx_buff)

        input = input_queue.get()
        if input_count % 50 == 0:
            print(f"sample {input_count}")

        start_time = time.time()
        print('edge device:')
        # Forward pass through the model
        try:
            out, ids, mask = models[0](input)
        except Exception as e:
            print(e)

        # if server idle
        print('outgoing queue size: ', outgoing_queue.qsize())
        while outgoing_queue.qsize() > 5:
            print('waiting...')
            time.sleep(1)

        if calculate_opt.incoming_count >= calculate_opt.outgoint_count and outgoing_queue.qsize() < 3:
            outgoing_queue.put([1, out, ids, mask])
            end_time = time.time()
            print('client computation time: ', end_time - start_time)
            calculate_opt.client_comp_statistics = (0, end_idx_buff, end_time - start_time)
            print('server idle!')
            continue

        for k in range(1, end_idx + 1):
            try:
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                if k == head_idx:
                    is_early_exit, lm_logits = early_exit_lm_head(lm_models, out)
                    print('is early: ', is_early_exit)
                    if is_early_exit:
                        break
            except Exception as e:
                print('oom!!!')
                is_oom = True

                end_idx = k - 1

                print('updated end index: ', end_idx)
                break

        end_time = time.time()
        print('client computation time: ', end_time - start_time)

        if not is_early_exit:
            cycle_count = cycle_count + 1
            input_count = input_count + 1

            outgoing_queue.put([end_idx + 1, out, ids, mask])
            print('outgoing queue PUT!')
            calculate_opt.client_comp_statistics = (end_idx, end_idx_buff, end_time - start_time)

            if is_oom:
                end_idx = math.ceil(end_idx / 2)
                is_oom = False

            if (input_count) % 1 == 0 and input_count < 5 and end_idx < max_layers:
                print('testing higher value(i<30)')
                calculate_opt.max_end_idx = end_idx
                end_idx = end_idx + 1

            if cycle_count == 2 and input_count > 5:
                print('testing lower value (i>30)')
                end_idx = max(0, end_idx - 2)

            if cycle_count > 2 and input_count >= 5  and end_idx < max_layers:
                print('testing higher value (i>30): ')
                calculate_opt.max_end_idx = end_idx
                end_idx = end_idx + 1

        #if (input_count) % 10 == 0:
        if len(calculate_opt.server_comp_statistics) >= 5:
            print(':))')
            end_idx, new_buff_idx = calculate_opt.calclate_opt()
            while new_buff_idx < end_idx_buff:
                models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)
            cycle_count = 0

        #if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
        if end_idx_buff < end_idx and end_idx_buff < max_layers:
            models, end_idx_buff = layer_reallocation(1, start_idx, end_idx_buff, max_layers, models)
        while end_idx_buff > end_idx + 3:  #remove buffer
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)
        gc.collect()
        torch.cuda.empty_cache()

    print('end T2...')



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

    max_layers = args.max_layers

    start_idx = args.start_idx
    end_idx_buff = args.end_idx_buff

    device = torch.device("cuda")
    head_idx = 2

    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx_buff, device)
    lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")
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
    nsamples = 50
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

    gc.collect()

    start_idx = 0
    end_idx = args.end_idx
    # Create and start threads
    thread1 = threading.Thread(target=task1_data_sending, args=[args])
    thread2 = threading.Thread(target=task2_computation, args=[models, lm_models, start_idx, end_idx, end_idx_buff, head_idx, max_layers, device])
    #thread3 = threading.Thread(target=task3_summerizing, args=[models, test_loader, bs, device])
    thread1.start()
    thread2.start()
    #thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    #thread3.join()

    print("Both tasks completed!")
