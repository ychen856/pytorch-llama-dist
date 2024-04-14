import asyncio
from http.server import HTTPServer

import aiohttp
from aiohttp import web
import json
from collections import deque

# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
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

import http_sender
from data import get_loaders
import torch.nn as nn
import safetensors
import http_receiver
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

from multiprocessing import set_start_method, Manager

import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from eval_sep_hf import get_eval_data
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml


# Global deque to store data from HTTP requests
data_queue = asyncio.Queue()
outgoing_queue = deque()


parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()


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

            models[j].to(device)

    '''for i in range(0, len(models)):
        model = models[i]
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)'''

    return models


# Function to handle HTTP POST requests
async def handle_post(request):
    print('hi')
    #data = await request.read()
    data = await request.content.read()
    print('data: ', data)
    # Add received data to the queue
    decrypt_data = pickle.loads(data)
    data_queue.put(decrypt_data)

    return web.json_response({'status': 'received'})


# Function to perform computation
async def compute_data(models, start_idx, end_idx, device):
    while True:
        print('hi')
        time.sleep(5)
        if not data_queue.empty():
            data = await data_queue.get()
            print('hii')

            # Perform computation
            out = data[0]
            ids = data[1]
            mask = data[2]
            # http_receiver.pop_incoming_queue()
            start_time = time.time()

            for k in range(start_idx, 33):
                k = k - start_idx
                start_time_sub = time.time()
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                end_time_sub = time.time()
                print(k, end_time_sub - start_time_sub)
                # print(k)
                # print('out: ', out)

            start_time_sub = time.time()
            lm_logits = models[33 - start_idx](out.last_hidden_state)
            end_time_sub = time.time()
            print('33:', end_time_sub - start_time_sub)

            start_time_sub = time.time()
            lm_logits = models[34 - start_idx](lm_logits)
            end_time_sub = time.time()
            print('34: ', end_time_sub - start_time_sub)

            print('lm_logits: ', lm_logits)


            print('output shape: ', lm_logits.shape)
            end_time = time.time()
            print('server computation time: ', end_time - start_time)
            print('computation finished!!')

            outgoing_queue.append(lm_logits)
            print('data store!!')

        # Sleep for a short duration to avoid high CPU usage
        await asyncio.sleep(0.01)


if __name__ == '__main__':
    set_start_method('spawn')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    start_idx = 5
    end_idx = 34
    # allow_cuda = False
    # device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    # Create an aiohttp web application
    app = web.Application()

    # Add a route for handling POST requests
    app.router.add_post('/', handle_post)

    # Start the aiohttp web server
    web.run_app(app, port=args.server_port)


    # Start the computation coroutine
    asyncio.run(compute_data(models, start_idx, end_idx, device))
