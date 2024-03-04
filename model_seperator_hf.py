# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.

from typing import Optional

import numpy as np
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import argparse
from data import get_loaders
import torch.nn as nn
import safetensors

from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

import sys

from eval import eval_ppl_sep_hf, eval_ppl_hf

import yaml

from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_linear, \
    LlamaForCausalLM_norm

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_nrp.yaml')
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


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    print('config type: ', args.config)
    torch.manual_seed(0)








    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    '''print(torch.cuda.memory_allocated())

    model = get_llm(args.ckpt_dir_hf, 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    print('model: ', model)

    print(torch.cuda.memory_allocated())
    torch.set_default_dtype(torch.float16)
    model2 = LlamaForCausalLM(model.config)
    model2.seqlen = 1024
    model2.load_state_dict(model.state_dict())
    torch.save(model2.state_dict(), args.ckpt_dir_hf + '/consolidated.00.pth')
    del model
    print(torch.cuda.memory_allocated())'''

    checkpoints = sorted(Path(args.ckpt_dir_hf).glob("*.pth"))
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    torch.set_default_dtype(torch.float16)
    model2 = LlamaForCausalLM(config)
    model2.seqlen = 1024
    model2.load_state_dict(checkpoint, strict=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    model2.cuda()
    print('model2: ', model2)

    '''print(torch.cuda.memory_allocated())
    for name, param in model2.named_parameters():
        if param.requires_grad:
            print(name, param.data)'''

    print("loading success")

    '''ppl = eval_ppl_hf(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")'''

    ppl = eval_ppl_hf(model2, tokenizer, device)
    print(f"ppl on wikitext {ppl}")

    model_embbding = LlamaForCausalLM_emb(config)
    model_embbding.model.embed_tokens = model2.model.embed_tokens
    torch.save(model_embbding.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.00.pth')

    model_b0 = LlamaForCausalLM_layer_0(config)
    model_b0.model.layers = model2.model.layers[0]
    torch.save(model_b0.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.01.pth')

    model_b1 = LlamaForCausalLM_layer_0(config)
    model_b1.model.layers = model2.model.layers[1]
    torch.save(model_b1.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.02.pth')

    model_b2 = LlamaForCausalLM_layer_0(config)
    model_b2.model.layers = model2.model.layers[2]
    torch.save(model_b2.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.03.pth')

    model_b3 = LlamaForCausalLM_layer_0(config)
    model_b3.model.layers = model2.model.layers[3]
    torch.save(model_b3.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.04.pth')

    model_b4 = LlamaForCausalLM_layer_0(config)
    model_b4.model.layers = model2.model.layers[4]
    torch.save(model_b4.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.05.pth')

    model_b5 = LlamaForCausalLM_layer_0(config)
    model_b5.model.layers = model2.model.layers[5]
    torch.save(model_b5.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.06.pth')

    model_b6 = LlamaForCausalLM_layer_0(config)
    model_b6.model.layers = model2.model.layers[6]
    torch.save(model_b6.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.07.pth')

    model_b7 = LlamaForCausalLM_layer_0(config)
    model_b7.model.layers = model2.model.layers[7]
    torch.save(model_b7.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.08.pth')

    model_b8 = LlamaForCausalLM_layer_0(config)
    model_b8.model.layers = model2.model.layers[8]
    torch.save(model_b8.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.09.pth')

    model_b9 = LlamaForCausalLM_layer_0(config)
    model_b9.model.layers = model2.model.layers[9]
    torch.save(model_b9.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.10.pth')

    model_b10 = LlamaForCausalLM_layer_0(config)
    model_b10.model.layers = model2.model.layers[10]
    torch.save(model_b10.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.11.pth')

    model_b11 = LlamaForCausalLM_layer_0(config)
    model_b11.model.layers = model2.model.layers[11]
    torch.save(model_b11.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.12.pth')

    model_b12 = LlamaForCausalLM_layer_0(config)
    model_b12.model.layers = model2.model.layers[12]
    torch.save(model_b12.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.13.pth')

    model_b13 = LlamaForCausalLM_layer_0(config)
    model_b13.model.layers = model2.model.layers[13]
    torch.save(model_b13.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.14.pth')

    model_b14 = LlamaForCausalLM_layer_0(config)
    model_b14.model.layers = model2.model.layers[14]
    torch.save(model_b14.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.15.pth')

    model_b15 = LlamaForCausalLM_layer_0(config)
    model_b15.model.layers = model2.model.layers[15]
    torch.save(model_b15.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.16.pth')

    model_b16 = LlamaForCausalLM_layer_0(config)
    model_b16.model.layers = model2.model.layers[16]
    torch.save(model_b16.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.17.pth')

    model_b17 = LlamaForCausalLM_layer_0(config)
    model_b17.model.layers = model2.model.layers[17]
    torch.save(model_b17.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.18.pth')

    model_b18 = LlamaForCausalLM_layer_0(config)
    model_b18.model.layers = model2.model.layers[18]
    torch.save(model_b18.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.19.pth')

    model_b19 = LlamaForCausalLM_layer_0(config)
    model_b19.model.layers = model2.model.layers[19]
    torch.save(model_b19.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.20.pth')

    model_b20 = LlamaForCausalLM_layer_0(config)
    model_b20.model.layers = model2.model.layers[20]
    torch.save(model_b20.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.21.pth')

    model_b21 = LlamaForCausalLM_layer_0(config)
    model_b21.model.layers = model2.model.layers[21]
    torch.save(model_b21.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.22.pth')

    model_b22 = LlamaForCausalLM_layer_0(config)
    model_b22.model.layers = model2.model.layers[22]
    torch.save(model_b22.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.23.pth')

    model_b23 = LlamaForCausalLM_layer_0(config)
    model_b23.model.layers = model2.model.layers[23]
    torch.save(model_b23.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.24.pth')

    model_b24 = LlamaForCausalLM_layer_0(config)
    model_b24.model.layers = model2.model.layers[24]
    torch.save(model_b24.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.25.pth')

    model_b25 = LlamaForCausalLM_layer_0(config)
    model_b25.model.layers = model2.model.layers[25]
    torch.save(model_b25.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.26.pth')

    model_b26 = LlamaForCausalLM_layer_0(config)
    model_b26.model.layers = model2.model.layers[26]
    torch.save(model_b26.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.27.pth')

    model_b27 = LlamaForCausalLM_layer_0(config)
    model_b27.model.layers = model2.model.layers[27]
    torch.save(model_b27.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.28.pth')

    model_b28 = LlamaForCausalLM_layer_0(config)
    model_b28.model.layers = model2.model.layers[28]
    torch.save(model_b28.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.29.pth')

    model_b29 = LlamaForCausalLM_layer_0(config)
    model_b29.model.layers = model2.model.layers[29]
    torch.save(model_b29.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.30.pth')

    model_b30 = LlamaForCausalLM_layer_0(config)
    model_b30.model.layers = model2.model.layers[30]
    torch.save(model_b30.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.31.pth')

    model_b31 = LlamaForCausalLM_layer_0(config)
    model_b31.model.layers = model2.model.layers[31]
    torch.save(model_b31.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.32.pth')

    model_norm = LlamaForCausalLM_norm(config)
    model_norm.model.norm = model2.model.norm
    torch.save(model_norm.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.33.pth')

    model_linear = LlamaForCausalLM_linear(config)
    model_linear.lm_head = model2.lm_head
    torch.save(model_linear.state_dict(), args.ckpt_dir_hf_sep + '/consolidated.34.pth')









