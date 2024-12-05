# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import math
import threading
from typing import Optional

import numpy as np
import torch
import time
from pathlib import Path
import argparse

import torch.nn as nn
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import sys

from eval import eval_ppl_sep_hf, eval_lm_head_ppl_sep_hf
from eval_sep_hf import get_eval_data
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
parser.add_argument('--head', type=int)
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

    #for early exit
    ckpt_path = checkpoints[-2]
    print(f'Loading checkpoint "{ckpt_path}"')
    checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))

    #for early exit
    ckpt_path = checkpoints[-1]
    print(f'Loading checkpoint "{ckpt_path}"')
    checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))

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

    # for early exit
    models.append((LlamaForCausalLM_norm(config)))
    models[end_idx + 1].load_state_dict(checkpoint_list[-2], strict=True)
    #models[end_idx + 1].cpu()
    models[end_idx + 1].to(device)

    models.append((LlamaForCausalLM_linear(config)))
    models[end_idx + 2].load_state_dict(checkpoint_list[-1], strict=True)
    #models[end_idx + 2].cpu()
    models[end_idx + 2].to(device)

    for i in range(0, len(models)):
        model = models[i]
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    return models


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    print('head:', args.head)
    start_idx = args.head
    end_idx = 10

    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
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
    #nsamples = 11
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    optimizer = AdamW(models[-1].parameters(), lr=5e-5)

    num_epochs = 20
    num_training_steps = num_epochs * nsamples
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    opt_ppl = np.inf
    models[-1].train()
    for epoch in range(num_epochs):
        nlls = []
        for i in tqdm(range(0, nsamples, bs)):
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
            inputs = inputs.reshape(j - i, seqlen)

            out, ids, mask = models[0](inputs)
            for k in range(1, len(models) - 2):
                start_time = time.time()
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)

            lm_logits = models[-2](out.last_hidden_state)
            lm_logits = models[-1](lm_logits)

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            #progress_bar.update(1)



            neg_log_likelihood = loss.float() * seqlen * (j - i)
            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)
            sys.stdout.flush()

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        if ppl.item() < opt_ppl:
            opt_ppl = ppl.item()
            #torch.save(models[-1].state_dict(), args.ckpt_dir_hf_sep + '/lm_head.10.pth')
            torch.save(models[-1].state_dict(), args.ckpt_dir_hf_sep + '/lm_head.'+args.head+'.pth')


        print('ppl: ', ppl.item())
        # Empty CUDA cache to save memory

    ppl = eval_ppl_sep_hf(models, tokenizer, device)
    print('eval ppl: ', ppl)








