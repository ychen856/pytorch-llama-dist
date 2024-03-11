# Import necessary modules
import time
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from data import get_loaders
import http_sender

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def get_eval_data(tokenizer):
    seqlen = 1024
    # Set dataset
    dataset = "wikitext2_hf"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer
    )

    return testloader


def eval_ppl_wikitext_sep_hf(models, testenc, args, start_idx = 0, end_idx = 34, bs=1, device=None):
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

        lm_logits = None
        if (start_idx == 0 and end_idx < 33):
            print('edge device:')
            # Forward pass through the model
            out, ids, mask = models[0](inputs)

            for k in range(1, len(models)):
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            http_sender.send_data(args.server_ip, out)

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

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

def eval_dist(models, testenc, args, start_idx = 0, end_idx = 34, bs=1, device=None):
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

        lm_logits = None
        if (start_idx == 0 and end_idx < 33):
            print('edge device:')
            # Forward pass through the model
            out, ids, mask = models[0](inputs)

            for k in range(1, len(models)):
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            http_sender.send_data(args.server_ip, out)

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

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()