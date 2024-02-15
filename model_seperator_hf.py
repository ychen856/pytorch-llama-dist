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

from eval import eval_ppl_sep_hf
from layerwrapper import WrappedGPT
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
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
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = 20
    return model

def get_llm2(checkpoints_dir, start_idx, end_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    models = []

    prev_time = time.time()
    checkpoint_list = []

    checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    for checkpoint in checkpoints:
        ckpt_path = checkpoint
        print(f'Loading checkpoint "{ckpt_path}"')

        checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
    prev_time = time.time()


    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)


    for i in range(start_idx, end_idx + 1):
        print('i', i)
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[0].model.embed_tokens.weight = nn.Parameter(checkpoint_list[0]['tok_embeddings.weight'])
            models[0].model.embed_tokens.type(torch.float16)
            models[0].to(device)

        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[33].model.norm.weight = nn.Parameter(checkpoint_list[33]['norm.weight'])
            models[33].model.norm.type(torch.float16)
            models[33].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[34].lm_head.weight = nn.Parameter(checkpoint_list[34]['output.weight'])
            models[34].lm_head.type(torch.float16)
            models[34].to(device)
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[i].model.layers.self_attn.q_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wq.weight'])
            models[i].model.layers.self_attn.k_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wk.weight'])
            models[i].model.layers.self_attn.v_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wv.weight'])
            models[i].model.layers.self_attn.o_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wo.weight'])
            models[i].model.layers.self_attn.q_proj.type(torch.float16)
            models[i].model.layers.self_attn.k_proj.type(torch.float16)
            models[i].model.layers.self_attn.v_proj.type(torch.float16)
            models[i].model.layers.self_attn.o_proj.type(torch.float16)

            models[i].model.layers.mlp.gate_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wq.weight'])
            models[i].model.layers.mlp.up_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wk.weight'])
            models[i].model.layers.mlp.down_proj.weight = nn.Parameter(checkpoint_list[i]['layers.attention.wv.weight'])
            models[i].model.layers.mlp.gate_proj.type(torch.float16)
            models[i].model.layers.mlp.up_proj.type(torch.float16)
            models[i].model.layers.mlp.down_proj.type(torch.float16)

            models[i].model.layers.input_layernorm.weight = nn.Parameter(checkpoint_list[i]['layers.attention_norm.weight'])
            models[i].model.layers.post_attention_layernorm.weight = nn.Parameter(checkpoint_list[i]['layers.ffn_norm.weight'])
            models[i].model.layers.input_layernorm.type(torch.float16)
            models[i].model.layers.post_attention_layernorm.type(torch.float16)
            models[i].to(device)


        print(f"Loaded state dict in {time.time() - prev_time:.2f}s")


    for i in range(0, len(models)):
        model = models[i]
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    return models


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    torch.manual_seed(0)

    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")

    #model = get_llm(args.ckpt_dir_hf, 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    print("loading success")



    models = get_llm2(args.ckpt_dir_sep, 0, 34, device, 'llm_weights')
    ppl = eval_ppl_sep_hf(models, tokenizer, device)
    print(f"ppl on wikitext {ppl}")
    prune_n = 0
    prune_m = 0

    #==========owl wanda=============
    all_layer_ratio = []

    use_cache = models[0].config.use_cache
    models[0].config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=0, seqlen=1024, tokenizer=tokenizer)
    print("dataset loading complete")

    inps, outs, attention_mask, position_ids = prepare_calibration_input(models, dataloader, device)
    print('inps: ', inps)
    print('outs: ', outs)
    print('attention_mask: ', attention_mask)
    print('position_ids: ', position_ids)

    for i in range(1, len(models) - 1):
        layer = models[i].model.layers
        subset = find_layers(layer)
        print('subset: ', subset)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                #if "OPT" in model.__class__.__name__:
                #    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                #else:
                #    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            activation_data = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                #if "OPT" in model.__class__.__name__:
                #    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                #else:
                #    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps



        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)

    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (
                1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2))

    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)

    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))

    print("after adjustment", all_layer_ratio)

    models[0].config.use_cache = use_cache
    torch.cuda.empty_cache()

    ############## prune

    use_cache = models[0].config.use_cache
    models[0].config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=0, seqlen=1024, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():

        #if "OPT" in model.__class__.__name__:

        #    inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        #else:

        #    inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        inps, outs, attention_mask, position_ids = prepare_calibration_input(models, dataloader, device)

    print("inps", inps)
    ''' if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers'''


    for i in range(1, len(models) - 1):
        layer = models[i].model.layers

        subset = find_layers(layer)

        '''if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)'''

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])


        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp


        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                '''if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]'''

                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            activation_data = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            layer_sparsity_ratio = 1 - all_layer_ratio[i - 1]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                ''' if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]'''

                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    models[0].config.use_cache = use_cache
    torch.cuda.empty_cache()

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(models)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl = eval_ppl_sep_hf(models, tokenizer, device)
    print(f"ppl on wikitext {ppl}")

    sys.stdout.flush()

    #=========owl wanda end===========
    '''#model_temp = LlamaForCausalLM(model.config)
    #print(model_temp)

    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)
    model_temp1 = LlamaForCausalLM_emb(config)
    print(model_temp1)
    model_temp2 = LlamaForCausalLM_layer_0(config)
    print(model_temp2)
    model_temp3 = LlamaForCausalLM_norm(config)
    print(model_temp3)

    wikisamples2 = ['Guard , office , and police duties .',
                    'Perhaps the most illuminating points of the above " Summary of Work " and those for following months are that the standard ammunition made was . " buck & ball " , indicating that the .69 caliber smoothbores and shotguns remained the predominant caliber weapon in use , and of this , nearly one sixth or more of all small arms ammunition was still for flintlock weapons , indicating that no less than a sixth of the Confederate troops in this vicinity were still armed with obsolete flintlock weapons .',
                    'The " Summaries of Work done at Little Rock Arsenal , C.S.A. " continue at about the same pace and scale from August 1862 until August 1863 . Appended to the " Summary " for August , 1863 is the ominous notation , " During the last week in the month , nearly all stores at the Arsenal have been packed and sent to Arkadelphia , in obedience to orders from Chief of Ordnance , District of Arkansas . " This then marks the beginning of the evacuation of ordnance activities from Little Rock , with the city being surrendered to the advancing Federal troops of Frederick Steele \'s Arkansas Expedition on September 11, 1863.',
                    'In 1864 , after Little Rock fell to the Union Army and the arsenal had been recaptured , General Fredrick Steele marched 8 @,@ 500 troops from the arsenal beginning the Camden Expedition .',
                    'The arsenal was briefly seized once more by Joseph Brooks loyalists during the Brooks @-@ Baxter War of 1874 .']

    testenc = tokenizer("\n\n".join(wikisamples2), return_tensors='pt')
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    print('testenc hf numel: ', testenc.numel())
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    nsamples = 5
    bs = 1
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model_temp1(inputs)
        print('logits1: ', lm_logits)

        lm_logits = model_temp2(lm_logits)
        print('logits2: ', lm_logits)

        lm_logits = model_temp3(lm_logits)
        print('logits3: ', lm_logits)
        # print('lm logits: ', lm_logits)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # print('shift logits: ', shift_logits)
        # print('shift labels: ', shift_labels)
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # print('loss fct shift logits: ', shift_logits.reshape(-1, shift_logits.size(-1)))
        # print('loss fct shift labels: ', shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

        # print ("nlls",nlls)
        sys.stdout.flush()

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    print('ppl: ', ppl.item())'''

    '''for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name)
    print('child')
    for child in list(model.model.children())[:-2]:
        print(child)'''

    #model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    #modules = list(model.model.children())[:-2]
    #model_embedding.save_pretrained(args.ckpt_dir_sep_hf, safe_serialization=True)

    '''model2 = get_llm(args.ckpt_dir_hf, 'llm_weights_emb')
    print(model2)'''

    '''for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name, param.data)'''

    '''with open(Path(args.ckpt_dir_hf) / "params.json", "r") as f:
        params = json.loads(f.read())

    ModelArgs = ModelArgs(
        max_seq_len=model.seqlen,
        max_batch_size=3,
        device=device,
        **params
    )'''

    '''model_embbding = Transformer_emb(ModelArgs)
    model_embbding.tok_embeddings = model.model.tok_embeddings
    torch.save(model_embbding.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.00.pth')

    model_b0 = Transformer_b0(model.args)
    model_b0.layers = model.model.layers[0]
    torch.save(model_b0.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.01.pth')

    model_b1 = Transformer_b1(model.args)
    model_b1.layers = model.model.layers[1]
    torch.save(model_b1.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.02.pth')

    model_b2 = Transformer_b2(model.args)
    model_b2.layers = model.model.layers[2]
    torch.save(model_b2.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.03.pth')

    model_b3 = Transformer_b3(model.args)
    model_b3.layers = model.model.layers[3]
    torch.save(model_b3.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.04.pth')

    model_b4 = Transformer_b4(model.args)
    model_b4.layers = model.model.layers[4]
    torch.save(model_b4.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.05.pth')

    model_b5 = Transformer_b5(model.args)
    model_b5.layers = model.model.layers[5]
    torch.save(model_b5.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.06.pth')

    model_b6 = Transformer_b6(model.args)
    model_b6.layers = model.model.layers[6]
    torch.save(model_b6.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.07.pth')

    model_b7 = Transformer_b7(model.args)
    model_b7.layers = model.model.layers[7]
    torch.save(model_b7.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.08.pth')

    model_b8 = Transformer_b8(model.args)
    model_b8.layers = model.model.layers[8]
    torch.save(model_b8.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.09.pth')

    model_b9 = Transformer_b9(model.args)
    model_b9.layers = model.model.layers[9]
    torch.save(model_b9.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.10.pth')

    model_b10 = Transformer_b10(model.args)
    model_b10.layers = model.model.layers[10]
    torch.save(model_b10.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.11.pth')

    model_b11 = Transformer_b11(model.args)
    model_b11.layers = model.model.layers[11]
    torch.save(model_b11.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.12.pth')

    model_b12 = Transformer_b12(model.args)
    model_b12.layers = model.model.layers[12]
    torch.save(model_b12.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.13.pth')

    model_b13 = Transformer_b13(model.args)
    model_b13.layers = model.model.layers[13]
    torch.save(model_b13.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.14.pth')

    model_b14 = Transformer_b14(model.args)
    model_b14.layers = model.model.layers[14]
    torch.save(model_b14.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.15.pth')

    model_b15 = Transformer_b15(model.args)
    model_b15.layers = model.model.layers[15]
    torch.save(model_b15.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.16.pth')

    model_b16 = Transformer_b16(model.args)
    model_b16.layers = model.model.layers[16]
    torch.save(model_b16.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.17.pth')

    model_b17 = Transformer_b17(model.args)
    model_b17.layers = model.model.layers[17]
    torch.save(model_b17.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.18.pth')

    model_b18 = Transformer_b18(model.args)
    model_b18.layers = model.model.layers[18]
    torch.save(model_b18.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.19.pth')

    model_b19 = Transformer_b19(model.args)
    model_b19.layers = model.model.layers[19]
    torch.save(model_b19.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.20.pth')

    model_b20 = Transformer_b20(model.args)
    model_b20.layers = model.model.layers[20]
    torch.save(model_b20.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.21.pth')

    model_b21 = Transformer_b21(model.args)
    model_b21.layers = model.model.layers[21]
    torch.save(model_b21.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.22.pth')

    model_b22 = Transformer_b22(model.args)
    model_b22.layers = model.model.layers[22]
    torch.save(model_b22.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.23.pth')

    model_b23 = Transformer_b23(model.args)
    model_b23.layers = model.model.layers[23]
    torch.save(model_b23.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.24.pth')

    model_b24 = Transformer_b24(model.args)
    model_b24.layers = model.model.layers[24]
    torch.save(model_b24.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.25.pth')

    model_b25 = Transformer_b25(model.args)
    model_b25.layers = model.model.layers[25]
    torch.save(model_b25.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.26.pth')

    model_b26 = Transformer_b26(model.args)
    model_b26.layers = model.model.layers[26]
    torch.save(model_b26.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.27.pth')

    model_b27 = Transformer_b27(model.args)
    model_b27.layers = model.model.layers[27]
    torch.save(model_b27.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.28.pth')

    model_b28 = Transformer_b28(model.args)
    model_b28.layers = model.model.layers[28]
    torch.save(model_b28.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.29.pth')

    model_b29 = Transformer_b29(model.args)
    model_b29.layers = model.model.layers[29]
    torch.save(model_b29.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.30.pth')

    model_b30 = Transformer_b30(model.args)
    model_b30.layers = model.model.layers[30]
    torch.save(model_b30.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.31.pth')

    model_b31 = Transformer_b31(model.args)
    model_b31.layers = model.model.layers[31]
    torch.save(model_b31.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.32.pth')

    model_norm = Transformer_norm(model.args)
    model_norm.norm = model.model.norm
    torch.save(model_norm.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.33.pth')

    model_linear = Transformer_linear(model.args)
    model_linear.output = model.model.output
    torch.save(model_linear.state_dict(), args.ckpt_dir_sep_hf + '/consolidated.34.pth')





'''