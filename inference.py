from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM
from tokenizers.implementations import BaseTokenizer
import argparse
import yaml

from datasets import load_dataset

#import llama_pruner
from model import ModelArgs, Transformer
from eval import *

#from prune_all import *

import torch.nn.utils.prune as prune
import torch.nn.functional as F


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

    model.seqlen = 1024
    return model

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)
        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model, tokenizer, model_args)

    #def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
    def text_completion(self, prompts, temperature = 0.6, top_p = 0.9, max_gen_len = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)

            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def eval(self, input):
        outputs = []
        max_gen_len = self.args.max_seq_len - 1
        max_prompt_len = max(len(prompt) for prompt in input)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        print('total len: ', total_len)
        cur_iterator = tqdm(range(1, total_len + 1), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logit = self.model.forward(input[:, cur_pos-1:cur_pos], cur_pos)
                outputs.append(logit)
        logits = torch.cat(outputs, dim=1)

        return logits

    def test(self, input):
        logit = self.model.forward(input, 0)




if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    '''model = LLaMA.build(
        checkpoints_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        load_model=True,
        max_seq_len=2048,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)'''

    wikisamples2 = ['Guard , office , and police duties .',
     'Perhaps the most illuminating points of the above " Summary of Work " and those for following months are that the standard ammunition made was . " buck & ball " , indicating that the .69 caliber smoothbores and shotguns remained the predominant caliber weapon in use , and of this , nearly one sixth or more of all small arms ammunition was still for flintlock weapons , indicating that no less than a sixth of the Confederate troops in this vicinity were still armed with obsolete flintlock weapons .',
     'The " Summaries of Work done at Little Rock Arsenal , C.S.A. " continue at about the same pace and scale from August 1862 until August 1863 . Appended to the " Summary " for August , 1863 is the ominous notation , " During the last week in the month , nearly all stores at the Arsenal have been packed and sent to Arkadelphia , in obedience to orders from Chief of Ordnance , District of Arkansas . " This then marks the beginning of the evacuation of ordnance activities from Little Rock , with the city being surrendered to the advancing Federal troops of Frederick Steele \'s Arkansas Expedition on September 11, 1863.',
     'In 1864 , after Little Rock fell to the Union Army and the arsenal had been recaptured , General Fredrick Steele marched 8 @,@ 500 troops from the arsenal beginning the Camden Expedition .',
     'The arsenal was briefly seized once more by Joseph Brooks loyalists during the Brooks @-@ Baxter War of 1874 .']

    wikisamples1 = ['= Valkyria Chronicles III =',
    'Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " .',
    'The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game \'s opening theme was sung by May \'n .',
    'It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game \'s expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .']
    wikisamples = ['= Valkyria Chronicles III =']


    ''' model = get_llm('/home/yichun/workspace/llama-2-7b-chat-hf/', 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained('/home/yichun/workspace/llama-2-7b-chat-hf', use_fast=False)
    testenc = tokenizer("\n\n".join(wikisamples1), return_tensors='pt')
    print('testenc hf: ', testenc)'''

    model = LLaMA.build(
        checkpoints_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        load_model=True,
        #max_seq_len=1024,

        max_seq_len=20,
        #max_batch_size=len(prompts),
        max_batch_size=1,
        device=device
    )

    # no pruning
    ppl = eval_ppl(model, model.tokenizer)
    print(f"ppl on wikitext {ppl}")

    #magnitude pruning
    '''for i in range(0, 32):
        for sub in ('wq', 'wk', 'wv', 'wo'):
            prune.random_unstructured(model.model.layers[i].attention._modules[sub], name="weight", amount=0.5)
            prune.remove(model.model.layers[i].attention._modules[sub], 'weight')
        for sub in ('w1', 'w2', 'w3'):
            prune.random_unstructured(model.model.layers[i].feed_forward._modules[sub], name="weight", amount=0.5)
            prune.remove(model.model.layers[i].feed_forward._modules[sub], 'weight')

    ppl = eval_ppl(model, model.tokenizer)
    print(f"ppl on wikitext {ppl}")'''

    #wanda pruning
    '''model = get_llm('/home/yichun/workspace/llama-2-7b-chat-hf/', 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained('/home/yichun/workspace/llama-2-7b-chat-hf', use_fast=False)
    testenc = tokenizer("\n\n".join(wikisamples2), return_tensors='pt')
    prune_wanda_outlier(args, model, tokenizer, device, prune_n=0, prune_m=0)'''


    #prune_mag_outlier(args, model, model.tokenizer, device, prune_n=0, prune_m=0)
    #model = get_llm('/home/yichun/workspace/llama-2-7b-chat-hf/', 'llm_weights')
    #tokenizer = LlamaTokenizer.from_pretrained('/home/yichun/workspace/llama-2-7b-chat-hf', use_fast=False)
    #ppl = eval_ppl_hf(model, tokenizer)
    #print(f"ppl on wikitext {ppl}")

    '''max_gen_len = model.args.max_seq_len
    # Convert each prompt into tokens
    prompt_tokens = [model.tokenizer.encode("\n\n".join(wikisamples2), out_type=int, add_bos=True, add_eos=False)]
    # Make sure the batch size is not too large
    batch_size = len(prompt_tokens)
    #assert batch_size <= model.args.max_batch_size, f"batch size must be less than or equal to {model.args.max_batch_size}"
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    # Make sure the prompt length is not larger than the maximum sequence length
    #assert max_prompt_len <= model.args.max_seq_len, f"prompt length must be less than or equal to {model.args.max_seq_len}"
    total_len = max(model.args.max_seq_len, max_gen_len + max_prompt_len)

    # Create the list that will contain the generated tokens, along with the initial prompt tokens
    pad_id = model.tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        # Populate the initial tokens with the prompt tokens
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    eos_reached = torch.tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
    testenc = tokens[prompt_tokens_mask.type(torch.bool)]
    testenc = torch.reshape(testenc, (1, testenc.numel()))


    # Get input IDs
    #testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.args.max_seq_len

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
        print('input: ', inputs.numel())

        lm_logits = model.eval(inputs)
        print('logits: ', lm_logits.numel())

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        print('shift logits: ', shift_logits)
        print('shift labels: ', shift_labels)
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()


        print('loss fct shift logits: ', shift_logits.reshape(-1, shift_logits.size(-1)))
        print('loss fct shift labels: ', shift_labels.reshape(-1))
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # print('loss fct shift logits: ', shift_logits.reshape(-1, shift_logits.size(-1)))
        # print('loss fct shift labels: ', shift_labels.reshape(-1))

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
    ppl = ppl.item()

    print(f"ppl on wikitext {ppl}")


    print('===============================')

    model = get_llm('/home/yichun/workspace/llama-2-7b-chat-hf/', 'llm_weights')
    tokenizer = LlamaTokenizer.from_pretrained('/home/yichun/workspace/llama-2-7b-chat-hf', use_fast=False)
    testenc = tokenizer("\n\n".join(wikisamples2), return_tensors='pt')
    #print('testenc hf: ', testenc)
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    nsamples = 1
    # Loop through each batch
    bs = 1
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)
        print('inputs hf:', inputs.numel())

        # Forward pass through the model
        lm_logits = model(inputs).logits
        print('lm logits hf: ', lm_logits.numel())
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        print('shift logits hf: ', shift_logits)
        print('shift labels hf: ', shift_labels)
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
    ppl = ppl.item()
    print(f"ppl on wikitext {ppl}")
'''


