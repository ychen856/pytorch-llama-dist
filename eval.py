# Import necessary modules
import time
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from data import get_loaders
from early_exit import early_exit_cuda_ppl_test

def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    testloader = get_loaders(
        dataset, seed=0, seqlen=model.args.max_seq_len, tokenizer=tokenizer
    )
    print('testenc: ', testloader)
    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl_hf(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2_hf"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext_hf(model, testloader, 1, device)
    return ppl

def eval_ppl_sep_hf(models, tokenizer, device=torch.device("cuda:0")):
    seqlen = 1024
    # Set dataset
    dataset = "wikitext2_hf"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        for i in range (1, 32):
            ppl = eval_ppl_wikitext_sep_hf(models, testloader, i, 1, device)
            print('i: ', i)
            print('ppl: ', ppl)
    return ppl

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):

    max_gen_len = model.args.max_seq_len - 1
    # Make sure the batch size is not too large
    batch_size = len(testenc)
    # assert batch_size <= model.args.max_batch_size, f"batch size must be less than or equal to {model.args.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in testenc)
    # Make sure the prompt length is not larger than the maximum sequence length
    # assert max_prompt_len <= model.args.max_seq_len, f"prompt length must be less than or equal to {model.args.max_seq_len}"
    total_len = max(model.args.max_seq_len, max_gen_len + max_prompt_len)

    # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = model.tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(testenc):
        # Populate the initial tokens with the prompt tokens
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    testenc = tokens[prompt_tokens_mask.type(torch.bool)]
    testenc = torch.reshape(testenc, (1, testenc.numel()))


    # Calculate number of samples
    nsamples = testenc.numel() // model.args.max_seq_len
    nsamples = 1
    print('testenc numel: ', testenc.numel())

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    nsamples = 1
    bs = 1
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.args.max_seq_len):(j * model.args.max_seq_len)].to(device)
        inputs = inputs.reshape(j - i, model.args.max_seq_len)

        lm_logits = model.eval(inputs)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.args.max_seq_len * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

        # print ("nlls",nlls)
        sys.stdout.flush()

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.args.max_seq_len))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_ppl_wikitext_hf(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    print('testenc hf numel: ', testenc.numel())
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
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))


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

    return ppl.item()

def eval_ppl_wikitext_sep_hf(models, testenc, splitting_point, bs=1, device=None):
    seqlen = 1024
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    nsamples = 5
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
        print('input: ', inputs)
        inputs = inputs.reshape(j - i, seqlen)
        print('inputs: ', inputs)
        print('inputs: ', inputs.shape)

        start_time = time.time()
        # Forward pass through the model
        out, ids, mask = models[0](inputs)
        end_time = time.time()
        print('0: ', end_time - start_time)
        #print('out: ', out)
        for k in range (1, len(models) - 2):
            print('k: ', k)
            start_time = time.time()
            out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            print('mask: ', mask)
            if k == splitting_point:
                out, ids, mask, pruned_data_idx_list, pruned_data_list = early_exit_cuda_ppl_test(models, out, ids, mask)
            end_time = time.time()
            print(k, end_time - start_time)
            #print('out: ', out)

        # recover data from the early exit
        for idx in pruned_data_idx_list:
            out.last_hidden_state[idx] = pruned_data_list[idx]


        start_time = time.time()
        lm_logits = models[33](out.last_hidden_state)
        end_time = time.time()
        print('33: ', end_time - start_time)
        #print('logit 33: ', lm_logits)

        start_time = time.time()
        lm_logits = models[34](lm_logits)
        end_time = time.time()
        print('34: ', end_time - start_time)
        print('logits: ', lm_logits)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        print('generated output: ', lm_logits)
        print('shift logits: ', shift_labels)
        print('shift lables: ', shift_labels)


        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j - i)


        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        sys.stdout.flush()

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()