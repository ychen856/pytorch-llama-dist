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

from early_exit import early_exit_lm_head
from eval import eval_ppl_sep_hf, eval_lm_head_ppl_sep_hf
from eval_sep_hf import get_eval_data, get_train_data
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

    decoder = FeatureDecoder(seq_len=seqlen)
    decoder.load_state_dict(checkpoint_list[0], strict=True)
    decoder.to(device)


    return decoder


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    print('topk: ', args.k)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    print('head:', args.head)
    start_idx = 0
    end_idx = 34
    #splitting_point = 2

    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    deEmbedding = FeatureDecoder(seq_len=128).to(device)


    print("loading success")
    #test_loader = get_eval_data(tokenizer)
    # Get input IDs
    #testenc = test_loader.input_ids

    # loading inputs data
    seqlen = 128

    trainenc = get_train_data(tokenizer, 128, 0.3)
    bs = 1

    # Calculate number of samples
    #nsamples = testenc.numel() // seqlen
    nsamples = len(trainenc)
    #nsamples = 11
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    scaler = GradScaler()
    optimizer = AdamW(deEmbedding.parameters(), lr=5e-5)

    num_epochs = 20
    num_training_steps = num_epochs * nsamples
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=200, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    opt_ppl = np.inf
    deEmbedding.train()
    for splitting_point in ([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]):
        torch.cuda.empty_cache()
        _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, splitting_point, device, cache_dir="llm_weights")
        nlls = []
        for epoch in range(num_epochs):
            for i in tqdm(range(0, nsamples, bs)):
                is_early_exit = False
                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                #inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                #inputs = inputs.reshape(j - i, seqlen)
                inputs = trainenc[i].to(device)
                lm_logits = None
                #print('inputs: ', inputs)
                #print('inputs size: ', inputs.shape)
                out, ids, mask = models[0](inputs)
                for k in range(1, len(models) - 2):
                    #print('k: ', k)
                    start_time = time.time()
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)


                    if k == splitting_point:
                        is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, splitting_point)

                        #MAX confidence
                        probs = lm_logits.softmax(dim=-1)  # [1, seq_len, vocab]
                        max_probs = probs.max(dim=-1).values  # [1, seq_len]
                        topk = random.choice([1, 3, 5, 8])
                        topk_indices = max_probs.topk(topk, dim=-1).indices  # [1, k]
                        selected_token_ids = max_probs[0, topk_indices[0].long()]  # [topk]
                        out.last_hidden_state = deEmbedding(selected_token_ids.unsqueeze(0).long())  # [1, topk]


                    #if is_early_exit:
                    #    break

                #if is_early_exit:
                #    continue


                lm_logits = models[-2](out.last_hidden_state)
                lm_logits = models[-1](lm_logits)

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]

                with autocast():
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                print(f"Epoch {epoch} | Split {splitting_point} | Loss: {loss.item():.4f}")


                #loss.backward()
                scaler.scale(loss).backward()

                # Check gradients BEFORE clipping
                invalid_grad = False
                for name, p in deEmbedding.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"❌ Invalid gradient in {name}")
                        invalid_grad = True
                        break

                if invalid_grad:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue  # skip this batch
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in deEmbedding.parameters() if p.grad is not None],
                        max_norm=1.0
                    )


                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                '''optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()'''
                progress_bar.update(1)



                neg_log_likelihood = loss.detach().float() * seqlen * (j - i)
                # Append to list of negative log likelihoods
                nlls.append(neg_log_likelihood)
                sys.stdout.flush()

                # Empty CUDA cache to save memory
                del out, lm_logits, loss, inputs
                torch.cuda.empty_cache()

            #break

            # Compute perplexity
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            if ppl.item() < opt_ppl:
                opt_ppl = ppl.item()
                #torch.save(models[-1].state_dict(), args.ckpt_dir_hf_sep + '/lm_head.10.pth')
                torch.save(deEmbedding.state_dict(), args.ckpt_dir_hf_sep + '/decoder.' + str(args.k) + '.pth')
                print(f"Saved new best model with PPL = {opt_ppl:.2f}")

        del lm_models
        #print('ppl: ', ppl.item())
        # Empty CUDA cache to save memory
        #torch.cuda.empty_cache()

    #ppl = eval_ppl_sep_hf(models, tokenizer, device)
    #print('eval ppl: ', ppl)

    '''#eval
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    deEmbedding = load_decoder(args.k, 512)


    print("loading success")

    # loading inputs data
    seqlen = 512
    test_loader = get_eval_data(tokenizer, seqlen)
    bs = 1
    # Get input IDs
    testenc = test_loader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    #nsamples = 11
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    scaler = GradScaler()
    optimizer = AdamW(models[-1].parameters(), lr=5e-5)

    progress_bar = tqdm(range(num_training_steps))

    opt_ppl = np.inf
    deEmbedding.train()

    for splitting_point in ([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]):
        print('splitting point: ', splitting_point)
        torch.cuda.empty_cache()

        with torch.no_grad():
            _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, splitting_point, device, cache_dir="llm_weights")
            nlls = []
            for i in tqdm(range(0, nsamples, bs)):
                is_early_exit = False
                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                inputs = inputs.reshape(j - i, seqlen)

                lm_logits = None

                # Start the model
                out, ids, mask = models[0](inputs)
                for k in range(1, len(models) - 2):
                    #print('k: ', k)
                    start_time = time.time()
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)


                    if k == splitting_point:
                        is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, splitting_point)

                        #MAX confidence
                        probs = lm_logits.softmax(dim=-1)  # [1, seq_len, vocab]
                        max_probs = probs.max(dim=-1).values  # [1, seq_len]
                        topk = random.choice([1, 3, 5, 8])
                        topk_indices = max_probs.topk(topk, dim=-1).indices  # [1, k]
                        selected_token_ids = max_probs[0, topk_indices[0].long()]  # [topk]
                        out.last_hidden_state = deEmbedding(selected_token_ids.unsqueeze(0).long())  # [1, topk]

                lm_logits = models[-2](out.last_hidden_state)
                lm_logits = models[-1](lm_logits)

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]

                with autocast():
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                print(f"Epoch {epoch} | Split {splitting_point} | Loss: {loss.item():.4f}")

                neg_log_likelihood = loss.detach().float() * seqlen * (j - i)
                # Append to list of negative log likelihoods
                nlls.append(neg_log_likelihood)
                sys.stdout.flush()

            # Compute perplexity
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

            print('ppl: ', ppl.item())
            # Empty CUDA cache to save memory
            torch.cuda.empty_cache()






    # train
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    deEmbedding = FeatureDecoder(seq_len=512).to(device)


    print("loading success")
    #test_loader = get_eval_data(tokenizer)
    # Get input IDs
    #testenc = test_loader.input_ids

    # loading inputs data
    seqlen = 512

    trainenc = get_train_data(tokenizer, seqlen, 0.7)
    bs = 1

    # Calculate number of samples
    #nsamples = testenc.numel() // seqlen
    nsamples = len(trainenc)
    #nsamples = 11
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    scaler = GradScaler()
    optimizer = AdamW(deEmbedding.parameters(), lr=5e-5)

    num_epochs = 20
    num_training_steps = num_epochs * nsamples
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=200, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    opt_ppl = np.inf
    deEmbedding.train()
    for splitting_point in ([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]):
        torch.cuda.empty_cache()
        _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, splitting_point, device, cache_dir="llm_weights")
        nlls = []
        for epoch in range(num_epochs):
            for i in tqdm(range(0, nsamples, bs)):
                is_early_exit = False
                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                #inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                #inputs = inputs.reshape(j - i, seqlen)
                inputs = trainenc[i]
                lm_logits = None
                #print('inputs: ', inputs)
                #print('inputs size: ', inputs.shape)
                out, ids, mask = models[0](inputs)
                for k in range(1, len(models) - 2):
                    #print('k: ', k)
                    start_time = time.time()
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)


                    if k == splitting_point:
                        is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, splitting_point)

                        #MAX confidence
                        probs = lm_logits.softmax(dim=-1)  # [1, seq_len, vocab]
                        max_probs = probs.max(dim=-1).values  # [1, seq_len]
                        topk = random.choice([1, 3, 5, 8])
                        topk_indices = max_probs.topk(topk, dim=-1).indices  # [1, k]
                        selected_token_ids = max_probs[0, topk_indices[0].long()]  # [topk]
                        out.last_hidden_state = deEmbedding(selected_token_ids.unsqueeze(0).long())  # [1, topk]

                        break

                with autocast():
                    loss_fct = nn.MSELoss()
                loss = loss_fct(out, lm_logits)
                print(f"Epoch {epoch} | Split {splitting_point} | Loss: {loss.item():.4f}")


                #loss.backward()
                scaler.scale(loss).backward()

                # Check gradients BEFORE clipping
                invalid_grad = False
                for name, p in deEmbedding.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"❌ Invalid gradient in {name}")
                        invalid_grad = True
                        break

                if invalid_grad:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue  # skip this batch
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in deEmbedding.parameters() if p.grad is not None],
                        max_norm=1.0
                    )


                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)



                neg_log_likelihood = loss.detach().float() * seqlen * (j - i)
                # Append to list of negative log likelihoods
                nlls.append(neg_log_likelihood)
                sys.stdout.flush()

            # Empty CUDA cache to save memory
            del out, lm_logits, loss, inputs
            torch.cuda.empty_cache()

            #break

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        if ppl.item() < opt_ppl:
            opt_ppl = ppl.item()
            #torch.save(models[-1].state_dict(), args.ckpt_dir_hf_sep + '/lm_head.10.pth')
            torch.save(deEmbedding.state_dict(), args.ckpt_dir_hf_sep + '/decoder.' + str(args.k) + '.pth')
            print(f"Saved new best model with PPL = {opt_ppl:.2f}")

        del lm_models
        #print('ppl: ', ppl.item())
        # Empty CUDA cache to save memory
        #torch.cuda.empty_cache()

    #ppl = eval_ppl_sep_hf(models, tokenizer, device)
    #print('eval ppl: ', ppl)'''
