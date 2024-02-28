from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

import argparse
import yaml

from model_dist import *
from eval import *

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()

class LLaMA:

    def __init__(self, model: [], tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, model: [], load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        checkpoint_list = []
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

            for checkpoint in checkpoints:
                ckpt_path = checkpoint
                print(f'Loading checkpoint "{ckpt_path}"')

                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
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

        i = 0
        for m in model:
            model[i] = m(model_args).to(device)
            print('checkpoint number: ', m.checkpoint)
            if load_model:
                model[i].load_state_dict(checkpoint_list[m.checkpoint], strict=True)
                print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

            i = i + 1


        '''i = 0
        for m in model:
            model[i] = m(model_args).to(device)
            if load_model:
                model[i].load_state_dict(checkpoint_list[i], strict=True)
                print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

            i = i + 1'''
        
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
                logits, freq = self.model[0].forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
                for i in range (1, len(self.model)):
                    logits = self.model[i].forward(logits, freq, cur_pos)
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
        temperature = args.temperature
        top_p = args.top_p
        outputs = []
        max_gen_len = self.args.max_seq_len - 1
        max_prompt_len = max(len(prompt) for prompt in input)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        cur_iterator = tqdm(range(1, total_len + 1), desc="Generating tokens")

        early_count = 0
        total_count = 0

        for cur_pos in cur_iterator:
            total_count = total_count + 1
            with torch.no_grad():
                logits, freq = self.model[0].forward(input[:, cur_pos-1:cur_pos], cur_pos)
                print('logits: ', logits)
                for i in range (1, len(self.model)):
                    logits = self.model[i].forward(logits, freq, cur_pos)
                    print(i)
                    print(logits)
                    if i == 21:
                        logits_norm = self.model[33].forward(logits, freq, cur_pos)
                        logits_linear = self.model[34].forward(logits_norm, freq, cur_pos)
                        probs = torch.softmax(logits_linear[:, -1] / temperature, dim=-1)
                        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                        if torch.max(probs_sort).item() > top_p:
                            logits = logits_linear
                            early_count = early_count + 1
                            break
                outputs.append(logits)

        logits = torch.cat(outputs, dim=1)
        print('early exit rate: ', early_count / total_count)
        return logits


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    allow_cuda = False
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
        checkpoints_dir=args.ckpt_dir_sep,
        tokenizer_path=args.tokenizer_path_sep,
        model = [Transformer_emb, Transformer_b0, Transformer_b1, Transformer_b2, Transformer_b3, Transformer_b4, Transformer_b5, Transformer_b6
                 , Transformer_b7, Transformer_b8, Transformer_b9, Transformer_b10, Transformer_b11, Transformer_b12, Transformer_b13, Transformer_b14
                 , Transformer_b15, Transformer_b16, Transformer_b17, Transformer_b18, Transformer_b19, Transformer_b20, Transformer_b21, Transformer_b22
                 , Transformer_b23, Transformer_b24, Transformer_b25, Transformer_b26, Transformer_b27, Transformer_b28, Transformer_b29, Transformer_b30
                 , Transformer_b31, Transformer_norm, Transformer_linear],
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)'''

    ################ early model ############
    model = LLaMA.build(
        checkpoints_dir=args.ckpt_dir_sep,
        tokenizer_path=args.tokenizer_path_sep,
        model=[Transformer_emb, Transformer_b0, Transformer_b1, Transformer_b2, Transformer_b3, Transformer_b4, Transformer_b5, Transformer_b6
            , Transformer_b7, Transformer_b8, Transformer_b9, Transformer_b10, Transformer_b11, Transformer_b12, Transformer_b13, Transformer_b14
            , Transformer_b15, Transformer_b16, Transformer_b17, Transformer_b18, Transformer_b19, Transformer_b20, Transformer_b21, Transformer_b22
            , Transformer_b23, Transformer_b24, Transformer_b25, Transformer_b26, Transformer_b27, Transformer_b28, Transformer_b29, Transformer_b30
            , Transformer_b31, Transformer_norm, Transformer_linear],
        load_model=True,
        max_seq_len=1024,
        max_batch_size=32,
        device=device
    )

    for i in range (0, len(model.model)):
        m = model.model[i]
        for name, param in m.named_parameters():
            if param.requires_grad:
                print(name, param.data)


    ppl = eval_ppl(model, model.tokenizer)
    print(f"ppl on wikitext after layer 29 {ppl}")