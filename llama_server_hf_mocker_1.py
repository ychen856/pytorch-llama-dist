# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import threading
from functools import partial
from typing import Optional

import os
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import argparse

import torch.nn as nn
import torch.nn.functional as F
import http_sender
from data import get_loaders
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig
from multiprocessing import Pool
from multiprocessing import set_start_method
import multiprocessing as mp

from multiprocessing import current_process
from threading import current_thread, Thread
import sys

from eval_sep_hf import get_eval_data
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue
import http_receiver
from time import sleep
from model import ModelArgs, Transformer
from model_dist import Transformer_emb, Transformer_b0, Transformer_b1, Transformer_b2, Transformer_b3, \
    Transformer_b4, Transformer_b5, Transformer_b6, Transformer_b7, Transformer_b8, Transformer_b9, Transformer_b10, Transformer_b11, \
    Transformer_b12, Transformer_b13, Transformer_b14, Transformer_b15, Transformer_b16, Transformer_b17, Transformer_b18, Transformer_b19, \
    Transformer_b20, Transformer_b21, Transformer_b22, Transformer_b23, Transformer_b24, Transformer_b25, Transformer_b26, Transformer_b27, \
    Transformer_b28, Transformer_b29, Transformer_b30, Transformer_b31, Transformer_norm, Transformer_linear
from prune_all import prepare_calibration_input_opt, prepare_calibration_input, find_layers, check_outlier_mean, \
    return_given_alpha, check_sparsity

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

incoming_queue = Queue()

def get_llm(model, cache_dir="llm_weights"):
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
        j = i - start_idx
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[j].load_state_dict(checkpoint_list[i], strict=True)
            #models[0].model.embed_tokens.weight = nn.Parameter(checkpoint_list[0]['model.embed_tokens.weight'])
            models[j].to(device)
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[j].load_state_dict(checkpoint_list[i], strict=True)
            #models[33].model.norm.weight = nn.Parameter(checkpoint_list[33]['model.norm.weight'])
            models[j].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[j].load_state_dict(checkpoint_list[i], strict=True)
            #models[34].lm_head.weight = nn.Parameter(checkpoint_list[34]['lm_head.weight'])
            models[j].to(device)
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[j].load_state_dict(checkpoint_list[i], strict=True)
            '''models[i].model.layers.self_attn.q_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.self_attn.q_proj.weight'])
            models[i].model.layers.self_attn.k_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.self_attn.k_proj.weight'])
            models[i].model.layers.self_attn.v_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.self_attn.v_proj.weight'])
            models[i].model.layers.self_attn.o_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.self_attn.o_proj.weight'])

            models[i].model.layers.mlp.gate_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.mlp.gate_proj.weight'])
            models[i].model.layers.mlp.up_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.mlp.up_proj.weight'])
            models[i].model.layers.mlp.down_proj.weight = nn.Parameter(checkpoint_list[i]['model.layers.mlp.down_proj.weight'])

            models[i].model.layers.input_layernorm.weight = nn.Parameter(checkpoint_list[i]['model.layers.input_layernorm.weight'])
            models[i].model.layers.post_attention_layernorm.weight = nn.Parameter(checkpoint_list[i]['model.layers.post_attention_layernorm.weight'])'''

            models[j].to(device)

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


def task1_data_receiving(args, inputs):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T1 do nothing!')

    while 1:
        http_receiver.run(port=args.server_port)
    ''' print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T1 do nothing!')
    for i in range(0, 5):
        sleep(0.1)
        incoming_queue.put(inputs[i])'''

def task2_computation(models, start_idx, end_idx, tokenizer, device, is_dummy=True):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T2 computaton...')
    while(1):
        print('start time: ', time.time())
        start_time_0 = time.time()
        if is_dummy:
            while incoming_queue.empty():
                print('wait...')
                time.sleep(0.01)

            input = incoming_queue.get()
        else:
            input = http_receiver.get_in_queue_data()

        #input = http_receiver.get_in_queue_data()
        print('start compute time: ', time.time())
        start_time = time.time()
        # Forward pass through the model
        if start_idx == 0:
            out, ids, mask = models[0](input)
        else:
            start_idx = input[0]
            out = input[1]
            ids = input[2]
            mask = input[3]

            zeros_row = torch.zeros((1, 1, out.last_hidden_state.size(2))).to(device)
            out.last_hidden_state = torch.cat((zeros_row, out.last_hidden_state), dim=1)

            zeros_tensor = torch.zeros_like(ids).to(device)
            ids = torch.cat((zeros_tensor, ids), dim=1)

            zeros_row = torch.zeros_like(mask[:, :, 0:1, :]).to(device)
            mask = torch.cat((zeros_row, mask), dim=3)

        print('out: ', out.last_hidden_state)
        print('ids: ', ids)
        print('mask: ', mask)

        end_time = time.time()
        print('0: ', end_time - start_time)
        start_comp_time = time.time()
        # print('out: ', out)
        for k in range(max(1, start_idx), len(models) - 2):
            start_time = time.time()
            out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            end_time = time.time()
            print(k, end_time - start_time)
            # print('out: ', out)
        total_comp_time = time.time() - start_comp_time
        print('out: ', out)
        print('end compute time: ', time.time())
        print('total computation time: ', total_comp_time)
        if is_dummy:
            break

        http_receiver.set_outgoing_queue([start_idx, total_comp_time])


        '''start_time = time.time()
        lm_logits = models[33](out.last_hidden_state)
        end_time = time.time()
        print('33: ', end_time - start_time)
        # print('logit 33: ', lm_logits)

        start_time = time.time()
        lm_logits = models[34](lm_logits)
        end_time = time.time()
        print('34: ', end_time - start_time)
        print('logits: ', lm_logits)
        print('logit size: ', lm_logits.size())

        shift_logits = lm_logits[:, :-1, :].contiguous()

        print('shift logits: ', shift_logits)

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        print('what is this? ', shift_logits.reshape(-1, shift_logits.size(-1)))

        preds = F.softmax(shift_logits.reshape(-1, shift_logits.size(-1))).argmax(dim=-1)
        print('preds: ', preds)
        #inps = torch.tensor([1]).cuda()
        #new_preds = torch.cat((inps, preds))
        #print('new preds: ', new_preds)
        reshaped_tensor = preds.view(1, -1)
        print('output size: ', reshaped_tensor.size())
        print('reshape: ', reshaped_tensor)
        print(tokenizer.batch_decode(reshaped_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False))'''



    print('round time: ', time.time() - start_time_0)

def init_worker(mps, fps, cut):
    global memorizedPaths, filepaths, cutoff
    global DG

    print("process initializing", mp.current_process())
    memorizedPaths, filepaths, cutoff = mps, fps, cut
    DG = 1##nx.read_gml("KeggComplete.gml", relabel = True)


if __name__ == '__main__':
    set_start_method('spawn')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)



    start_idx_buff = 0
    end_idx = 34
    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx_buff, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    inputs = get_dataset(tokenizer)
    print("loading success")
    # Create and start threads

    start_idx = 0
    end_idx = 34
    incoming_queue.put(inputs[0])
    thread2 = threading.Thread(target=task2_computation, args=[models, start_idx, end_idx, tokenizer, device, True])

    thread2.start()

    # Wait for both threads to finish (optional)
    thread2.join()

    start_time = time.time()
    start_idx = 5
    end_idx = 34
    # task2_computation(models, start_idx, end_idx, tokenizer, device, inputs)
    thread1 = threading.Thread(target=task1_data_receiving, args=[args, inputs])
    thread2 = threading.Thread(target=task2_computation, args=[models, start_idx, end_idx, tokenizer, device, False])

    thread1.start()
    thread2.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    print('total_time: ', time.time() - start_time)
