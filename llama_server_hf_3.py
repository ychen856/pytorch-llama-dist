import asyncio
import aiohttp
from aiohttp import web
import torch
import torch.nn as nn
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
import argparse
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
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

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

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

'''# Define a simple MLP model for demonstration
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the pretrained model
model = MLP(input_dim=10, hidden_dim=20, output_dim=1)  # Replace with your actual model
#model.load_state_dict(torch.load('model.pth'))  # Load your pretrained model
model.eval()'''


# Function to handle incoming HTTP POST requests
async def handle_request(request):
    global model
    data = await request.content.read()
    #inputs = torch.tensor(data['inputs'], dtype=torch.float32)

    # Move inputs and model to GPU
    #inputs = inputs.cuda()
    #model = model.cuda()

    # Perform inference
    with torch.no_grad():
        #outputs = model(inputs)
        out = data[0]
        ids = data[1]
        mask = data[2]
        # http_receiver.pop_incoming_queue()
        start_time = time.time()

        for k in range(start_idx, 33):
            k = k - start_idx
            start_time_sub = time.time()
            out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
            end_time_sub = time.time()
            print(k, end_time_sub - start_time_sub)
            # print(k)
            # print('out: ', out)

        start_time_sub = time.time()
        lm_logits = models[33 - start_idx](out.last_hidden_state)
        end_time_sub = time.time()
        print('33:', end_time_sub - start_time_sub)

        start_time_sub = time.time()
        lm_logits = models[34 - start_idx](lm_logits)
        end_time_sub = time.time()
        print('34: ', end_time_sub - start_time_sub)

        print('lm_logits: ', lm_logits)


        print('output shape: ', lm_logits.shape)
        end_time = time.time()
        print('server computation time: ', end_time - start_time)
        print('computation finished!!')

        #http_receiver.set_outgoing_queue(lm_logits)
        print('data store!!')

    # Move outputs back to CPU
    #outputs = outputs.cpu().numpy().tolist()

    #print(f"Inputs: {data['inputs']}")
    #print(f"Outputs: {outputs}")

    return web.json_response({'outputs': lm_logits})


start_idx = 0
end_idx = 34
device = torch.device("cuda")
models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

# Main function to setup and run the web server
async def main():
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)

    global start_idx
    global end_idx
    start_idx = 5
    end_idx = 34

    '''device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)'''

    # model = get_llm(args.ckpt_dir_hf, 'llm_weights')
    # tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    print("loading success")

    # Create an application and add route for handling HTTP POST requests
    app = web.Application()
    app.router.add_post('/', handle_request)

    # Run the web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '', args.server_port)
    print("Server started on http://localhost:8080")
    await site.start()

    await asyncio.Future()


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
