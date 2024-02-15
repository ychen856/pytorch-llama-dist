# Import necessary modules
import time
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from data import get_loaders

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
        ppl = eval_ppl_wikitext_sep_hf(models, testloader, 1, device)
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
    print('testenc numel: ', testenc.numel())

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




    '''# Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    #nsamples = testenc.numel() // model.seqlen
    nsamples = testenc.numel() // model.args.max_seq_len

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.args.max_seq_len):(j * model.args.max_seq_len)].to(device)
        inputs = inputs.reshape(j-i, model.args.max_seq_len)

        # Forward pass through the model
        #lm_logits = model(inputs).logits
        lm_logits = model(inputs, 0)
        print('inputs: ', inputs)
        print('logits: ', lm_logits)

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.args.max_seq_len * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.args.max_seq_len))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()'''


def eval_ppl_wikitext_hf(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    print('testenc hf numel: ', testenc.numel())
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    nsamples = 5
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
        #print('lm logits: ', lm_logits)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        #print('shift logits: ', shift_logits)
        #print('shift labels: ', shift_labels)
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        #print('loss fct shift logits: ', shift_logits.reshape(-1, shift_logits.size(-1)))
        #print('loss fct shift labels: ', shift_labels.reshape(-1))

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

def eval_ppl_wikitext_sep_hf(models, testenc, bs=1, device=None):
    seqlen = 1024
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
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

        # Forward pass through the model
        out, ids, mask = models[0](inputs)
        for i in range (1, len(models) - 2):
            out = models[i](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            mask = out.attentions
        print('out: ', out)
        print('model 33', models[33])
        lm_logits = models[33](out.last_hidden_state)
        lm_logits = models[34](lm_logits)
        #lm_logits = models[0](inputs).logits
        #print('lm logits: ', lm_logits)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        print('shift logits: ', shift_logits)
        print('shift labels: ', shift_labels)


        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

        # print ("nlls",nlls)
        sys.stdout.flush()

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()