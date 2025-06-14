# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import math
import random
import threading
from typing import Optional

import numpy as np
import torch
import time
from pathlib import Path
import argparse

from natsort import natsorted
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import sys
import gc
from early_exit import early_exit_lm_head
from eval import eval_ppl_sep_hf, eval_lm_head_ppl_sep_hf
from eval_sep_hf import get_eval_data, get_train_data
from feature_encoder import FlexibleTopKEncoder
from feature_decoder import FlexibleDecoder
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
import copy
from feature_decoder import *
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
parser.add_argument('--head', type=int)
parser.add_argument('--k', type=int)
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

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

    '''if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)'''

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
    print('zzzzzzzzzzz', checkpoints)
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"


    for i in range(0, len(checkpoints)):
        if i == 0 or i == lm_head_idx:
            ckpt_path = checkpoints[i]
            print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))



    '''if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)'''
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

def load_decoder(checkpoints_dir, k, seqlen=1024):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("decoder." + str(args.k) + ".pth"))
    checkpoints = natsorted(checkpoints)

    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    ckpt_path = checkpoints[0]
    print(f'Loading checkpoint "{ckpt_path}"')

    checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))

    '''if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)'''

    decoder = FlexibleDecoder(seq_len=seqlen)
    decoder.load_state_dict(checkpoint_list[0], strict=True)
    decoder.to(device)


    return decoder

if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    args.k = int(args.k)

    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda")

    seqlen = 1024
    bs = 1

    possible_ks = [32, 64, 128]
    #possible_ks = [64]
    possible_bottleneck_dims = [64, 128, 256, 512]

    models = load_model(args.ckpt_dir_hf_sep, 0, 34, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    encoder = FlexibleTopKEncoder().to(device)
    decoder = FlexibleDecoder(seq_len=seqlen).to(device)
    trainenc = get_train_data(tokenizer, seqlen, 0.7)


    nsamples = len(trainenc)

    #////////////
    scaler = GradScaler()
    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-5,
        weight_decay=0.01
    )
    opt_ppl = float('inf')
    num_epochs = 20

    for i in range(0, len(models)):
        models[i].eval()
        for p in models[i].parameters():
            p.requires_grad = False

    decoder.train()
    encoder.train()

    # --- Training loop with decoder-only training ---
    for splitting_point in [1, 2, 4, 6, 8]:
        torch.cuda.empty_cache()
        _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, splitting_point, device, cache_dir="llm_weights")
        for i in range(0, len(lm_models)):
            lm_models[i].eval()
            for p in lm_models[i].parameters():
                p.requires_grad = False
        nlls = []
        for epoch in range(num_epochs):
            total_loss = 0

            for i in range(0, len(trainenc), bs):
                j = min(i + bs, nsamples)
                optimizer.zero_grad(set_to_none=True)
                if args.k == 0:
                    top_k = random.choice(possible_ks)
                else:
                    top_k = args.k

                #top_k = 64
                print('top k: ', top_k)
                bottleneck_dim = random.choice(possible_bottleneck_dims)

                inputs = trainenc[i].to(device)

                with torch.no_grad():
                    out, ids, mask = models[0](inputs)
                    for k in range(1, splitting_point + 1):
                        out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)

                        if k == splitting_point:
                            is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, splitting_point)
                            # Step 1: compute confidence + top-k
                            probs = torch.softmax(lm_logits, dim=-1)
                            conf = probs.max(dim=-1).values  # [B, 1024]
                            topk_vals, topk_idx = conf.topk(top_k, dim=1)
                            B, _, V = lm_logits.shape
                            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, V)
                            topk_logits = torch.gather(lm_logits, dim=1, index=topk_idx_exp)

                z = encoder(topk_logits, bottleneck_dim=bottleneck_dim)  # [B, k, bottleneck_dim]
                recon_hidden = decoder(z, topk_idx, bottleneck_dim=bottleneck_dim)  # [B, 1024, H]

                print('decoder output size: ', recon_hidden.shape)

                # Check decoder output
                if torch.isnan(recon_hidden).any() or torch.isinf(recon_hidden).any():
                    print("❌ decoder output contains NaN or Inf")
                    continue
                max_val = recon_hidden.abs().max()
                if max_val > 1e4:
                    print(f"⚠️ decoder output too large: max={max_val.item():.2e}")
                    continue


                out, ids, mask = models[splitting_point + 1](recon_hidden, position_ids=ids, attention_mask=mask)
                for k in range(splitting_point + 2, len(models) - 2):
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)

                lm_logits = models[-2](out.last_hidden_state)
                lm_logits = models[-1](lm_logits)

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    max_norm=1.0
                )

                optimizer.step()

                nlls.append(loss.detach().float())
                print(f"Epoch {epoch} | Split {splitting_point} | Loss: {loss.item():.4f}")

                torch.cuda.empty_cache()

            ppl = torch.exp(torch.stack(nlls).mean())
            if ppl.item() < opt_ppl:
                opt_ppl = ppl.item()
                torch.save(encoder.state_dict(), args.ckpt_dir_hf_sep + f"/encoder." + str(args.k) + ".pth")
                torch.save(decoder.state_dict(), args.ckpt_dir_hf_sep + f"/decoder." + str(args.k) + ".pth")
                print(f"✅ Saved best model with PPL = {opt_ppl:.2f}")

