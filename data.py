# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(seqlen, tokenizer, device=torch.device("cuda:0")):
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    traindata = load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
    testdata = load_dataset(path="wikitext", name="wikitext-103-v1", split="test")

    # Convert each prompt into tokens
    prompt_tokens = [tokenizer.encode("\n\n".join(testdata['text']), out_type=int, add_bos=True, add_eos=False)]
    batch_size = len(prompt_tokens)
    # assert batch_size <= model.args.max_batch_size, f"batch size must be less than or equal to {model.args.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    # Make sure the prompt length is not larger than the maximum sequence length
    # assert max_prompt_len <= model.args.max_seq_len, f"prompt length must be less than or equal to {model.args.max_seq_len}"
    total_len = max(seqlen, seqlen + max_prompt_len)

    # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        # Populate the initial tokens with the prompt tokens
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise

    testenc = tokens[prompt_tokens_mask.type(torch.bool)]
    testenc = torch.reshape(testenc, (1, testenc.numel()))

    return testenc

# Load and process wikitext2 dataset
def get_wikitext2_hf(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    #traindata = load_dataset('/home/yichun/workspace/wikitext', 'wikitext-2-raw-v1', split='train')
    #testdata = load_dataset('/home/yichun/workspace/wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
    #testdata = load_dataset(path="wikitext", name="wikitext-103-v1", split="test")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    #traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    #valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2_hf' in name:
        return get_wikitext2_hf(nsamples, seed, seqlen, tokenizer)
    if 'wikitext2' in name:
        return get_wikitext2(seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)