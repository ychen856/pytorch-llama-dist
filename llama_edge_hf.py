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

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()


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



def task1_data_receiving(args):
    http_receiver.run(server_ip=args.client_ip, port=args.client_port)

def task2_computation(models, test_loader, bs, start_idx, end_idx, device):
    seqlen = 1024
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

        start_time = time.time()
        if (start_idx == 0 and end_idx < 33):
            print('edge device:')
            start_time_sub = time.time()
            # Forward pass through the model
            out, ids, mask = models[0](inputs)
            end_time_sub = time.time()
            print('0: ', end_time_sub - start_time_sub)
            for k in range(1, len(models)):
                start_time_sub = time.time()
                print('start time: ', start_time_sub)
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                end_time_sub = time.time()
                print('end time: ', end_time_sub)
                print(k, end_time_sub - start_time_sub)

            end_time = time.time()
            print('client computation time: ', end_time - start_time)

            http_sender.send_data(args.server_ip, args.server_port, [out, ids, mask])

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
    print('YAY')

def task3_summerizing(models, test_loader, bs, device):
    seqlen = 1024
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

        '''while 1:
            lm_logits = http_sender.get_queue_data()

            if len(lm_logits) > 0:
                #http_sender.pop_incoming_queue()
                break'''

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
    print('ppl: ', ppl.item())


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)


    start_idx = 0
    end_idx = 4
    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)


    print("loading success")
    test_loader = get_eval_data(tokenizer)
    bs = 1

    # Create and start threads
    #thread1 = threading.Thread(target=task1_data_receiving, args=[args])
    thread2 = threading.Thread(target=task2_computation, args=[models, test_loader, bs, start_idx, end_idx, device])
    thread3 = threading.Thread(target=task3_summerizing, args=[models, test_loader, bs, device])
    #thread1.start()
    thread2.start()
    thread3.start()

    # Wait for both threads to finish (optional)
    #thread1.join()
    thread2.join()
    thread3.join()

    print("Both tasks completed!")






    #ppl = eval_ppl_sep_hf(models, tokenizer, args, start_idx, end_idx, device)
    #print(f"ppl on wikitext {ppl}")


    '''models = get_llm2(args.ckpt_dir_sep, 0, 34, device, 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    ppl = eval_ppl_sep_hf(models, tokenizer, device)
    print(f"ppl on wikitext {ppl}")'''