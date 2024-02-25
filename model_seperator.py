# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.

from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import argparse
import yaml
from model import ModelArgs, Transformer
from model_dist import Transformer_emb, Transformer_b0, Transformer_b1, Transformer_b2, Transformer_b3, \
    Transformer_b4, Transformer_b5, Transformer_b6, Transformer_b7, Transformer_b8, Transformer_b9, Transformer_b10, Transformer_b11, \
    Transformer_b12, Transformer_b13, Transformer_b14, Transformer_b15, Transformer_b16, Transformer_b17, Transformer_b18, Transformer_b19, \
    Transformer_b20, Transformer_b21, Transformer_b22, Transformer_b23, Transformer_b24, Transformer_b25, Transformer_b26, Transformer_b27, \
    Transformer_b28, Transformer_b29, Transformer_b30, Transformer_b31, Transformer_norm, Transformer_linear

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cuda")
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

    model = LLaMA.build(
        checkpoints_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        load_model=True,
        max_seq_len=1024,
        #max_batch_size=len(prompts),
        max_batch_size = 3,
        device=device
    )


    print("loading success")

    '''for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name, param.data)'''


    model_embbding = Transformer_emb(model.args)
    model_embbding.tok_embeddings = model.model.tok_embeddings
    torch.save(model_embbding.state_dict(), args.ckpt_dir_sep + '/consolidated.00.pth')

    model_b0 = Transformer_b0(model.args)
    model_b0.layers = model.model.layers[0]
    torch.save(model_b0.state_dict(), args.ckpt_dir_sep + '/consolidated.01.pth')

    model_b1 = Transformer_b1(model.args)
    model_b1.layers = model.model.layers[1]
    torch.save(model_b1.state_dict(), args.ckpt_dir_sep + '/consolidated.02.pth')

    model_b2 = Transformer_b2(model.args)
    model_b2.layers = model.model.layers[2]
    torch.save(model_b2.state_dict(), args.ckpt_dir_sep + '/consolidated.03.pth')

    model_b3 = Transformer_b3(model.args)
    model_b3.layers = model.model.layers[3]
    torch.save(model_b3.state_dict(), args.ckpt_dir_sep + '/consolidated.04.pth')

    model_b4 = Transformer_b4(model.args)
    model_b4.layers = model.model.layers[4]
    torch.save(model_b4.state_dict(), args.ckpt_dir_sep + '/consolidated.05.pth')

    model_b5 = Transformer_b5(model.args)
    model_b5.layers = model.model.layers[5]
    torch.save(model_b5.state_dict(), args.ckpt_dir_sep + '/consolidated.06.pth')

    model_b6 = Transformer_b6(model.args)
    model_b6.layers = model.model.layers[6]
    torch.save(model_b6.state_dict(), args.ckpt_dir_sep + '/consolidated.07.pth')

    model_b7 = Transformer_b7(model.args)
    model_b7.layers = model.model.layers[7]
    torch.save(model_b7.state_dict(), args.ckpt_dir_sep + '/consolidated.08.pth')

    model_b8 = Transformer_b8(model.args)
    model_b8.layers = model.model.layers[8]
    torch.save(model_b8.state_dict(), args.ckpt_dir_sep + '/consolidated.09.pth')

    model_b9 = Transformer_b9(model.args)
    model_b9.layers = model.model.layers[9]
    torch.save(model_b9.state_dict(), args.ckpt_dir_sep + '/consolidated.10.pth')

    model_b10 = Transformer_b10(model.args)
    model_b10.layers = model.model.layers[10]
    torch.save(model_b10.state_dict(), args.ckpt_dir_sep + '/consolidated.11.pth')

    model_b11 = Transformer_b11(model.args)
    model_b11.layers = model.model.layers[11]
    torch.save(model_b11.state_dict(), args.ckpt_dir_sep + '/consolidated.12.pth')

    model_b12 = Transformer_b12(model.args)
    model_b12.layers = model.model.layers[12]
    torch.save(model_b12.state_dict(), args.ckpt_dir_sep + '/consolidated.13.pth')

    model_b13 = Transformer_b13(model.args)
    model_b13.layers = model.model.layers[13]
    torch.save(model_b13.state_dict(), args.ckpt_dir_sep + '/consolidated.14.pth')

    model_b14 = Transformer_b14(model.args)
    model_b14.layers = model.model.layers[14]
    torch.save(model_b14.state_dict(), args.ckpt_dir_sep + '/consolidated.15.pth')

    model_b15 = Transformer_b15(model.args)
    model_b15.layers = model.model.layers[15]
    torch.save(model_b15.state_dict(), args.ckpt_dir_sep + '/consolidated.16.pth')

    model_b16 = Transformer_b16(model.args)
    model_b16.layers = model.model.layers[16]
    torch.save(model_b16.state_dict(), args.ckpt_dir_sep + '/consolidated.17.pth')

    model_b17 = Transformer_b17(model.args)
    model_b17.layers = model.model.layers[17]
    torch.save(model_b17.state_dict(), args.ckpt_dir_sep + '/consolidated.18.pth')

    model_b18 = Transformer_b18(model.args)
    model_b18.layers = model.model.layers[18]
    torch.save(model_b18.state_dict(), args.ckpt_dir_sep + '/consolidated.19.pth')

    model_b19 = Transformer_b19(model.args)
    model_b19.layers = model.model.layers[19]
    torch.save(model_b19.state_dict(), args.ckpt_dir_sep + '/consolidated.20.pth')

    model_b20 = Transformer_b20(model.args)
    model_b20.layers = model.model.layers[20]
    torch.save(model_b20.state_dict(), args.ckpt_dir_sep + '/consolidated.21.pth')

    model_b21 = Transformer_b21(model.args)
    model_b21.layers = model.model.layers[21]
    torch.save(model_b21.state_dict(), args.ckpt_dir_sep + '/consolidated.22.pth')

    model_b22 = Transformer_b22(model.args)
    model_b22.layers = model.model.layers[22]
    torch.save(model_b22.state_dict(), args.ckpt_dir_sep + '/consolidated.23.pth')

    model_b23 = Transformer_b23(model.args)
    model_b23.layers = model.model.layers[23]
    torch.save(model_b23.state_dict(), args.ckpt_dir_sep + '/consolidated.24.pth')

    model_b24 = Transformer_b24(model.args)
    model_b24.layers = model.model.layers[24]
    torch.save(model_b24.state_dict(), args.ckpt_dir_sep + '/consolidated.25.pth')

    model_b25 = Transformer_b25(model.args)
    model_b25.layers = model.model.layers[25]
    torch.save(model_b25.state_dict(), args.ckpt_dir_sep + '/consolidated.26.pth')

    model_b26 = Transformer_b26(model.args)
    model_b26.layers = model.model.layers[26]
    torch.save(model_b26.state_dict(), args.ckpt_dir_sep + '/consolidated.27.pth')

    model_b27 = Transformer_b27(model.args)
    model_b27.layers = model.model.layers[27]
    torch.save(model_b27.state_dict(), args.ckpt_dir_sep + '/consolidated.28.pth')

    model_b28 = Transformer_b28(model.args)
    model_b28.layers = model.model.layers[28]
    torch.save(model_b28.state_dict(), args.ckpt_dir_sep + '/consolidated.29.pth')

    model_b29 = Transformer_b29(model.args)
    model_b29.layers = model.model.layers[29]
    torch.save(model_b29.state_dict(), args.ckpt_dir_sep + '/consolidated.30.pth')

    model_b30 = Transformer_b30(model.args)
    model_b30.layers = model.model.layers[30]
    torch.save(model_b30.state_dict(), args.ckpt_dir_sep + '/consolidated.31.pth')

    model_b31 = Transformer_b31(model.args)
    model_b31.layers = model.model.layers[31]
    torch.save(model_b31.state_dict(), args.ckpt_dir_sep + '/consolidated.32.pth')

    model_norm = Transformer_norm(model.args)
    model_norm.norm = model.model.norm
    torch.save(model_norm.state_dict(), args.ckpt_dir_sep + '/consolidated.33.pth')

    model_linear = Transformer_linear(model.args)
    model_linear.output = model.model.output
    torch.save(model_linear.state_dict(), args.ckpt_dir_sep + '/consolidated.34.pth')





