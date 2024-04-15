import queue
import threading
# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import threading
import torch
import time
from pathlib import Path
import argparse
import pickle
import http_receiver
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig
from http.server import BaseHTTPRequestHandler, HTTPServer
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

# Create a queue to store HTTP request data
data_queue = queue.Queue()
outgoing_queue = queue.Queue()


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

        print(next(models[j].parameters()).device)

    return models

# HTTP Server class to receive messages
class HTTPRequestHandler(BaseHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        self.end_headers()

    def do_POST(self):
        print('receive POST:')
        start_time = time.time()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self._set_response()
        print('length: ', content_length)
        # print(post_data)

        decrypt_data = pickle.loads(post_data)
        print(decrypt_data)

        data_queue.put(decrypt_data)
        # incoming_queue.append(decrypt_data)
        end_time = time.time()
        print('server receiving time: ', end_time - start_time)
        '''# Process the received data here:
        self.send_response(200)
        self.end_headers()
        newx = pickle.dumps('Data received successfully!')
        self.wfile.write(newx)'''

        self.return_message()

    def return_message(self):
        '''outgoing_data = []
        while 1:
            while not outgoing_queue.empty():
                outgoing_data = outgoing_queue.get()
            if len(outgoing_data) > 0:
                break'''

        while data_queue.empty():
            time.sleep(1.5)

        # Process the received data here:
        start_time = time.time()
        # Process the received data here:
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        self.end_headers()

        newx = pickle.dumps(outgoing_queue.get())
        #print('sent data: ', newx)
        self.wfile.write(newx)
        #outgoing_queue.pop(0)
        end_time = time.time()
        print('server sending time: ', end_time - start_time)
        print('end response')

        '''# Process the received data here:
        self.send_response(200)
        self.end_headers()
        newx = pickle.dumps('Data received successfully!')
        self.wfile.write(newx)'''

# Function to process data from the queue
def process_data(models, start_idx, end_idx, device):
    while True:
        if not data_queue.empty():
            data = data_queue.get()
            print('data: ', data)
            out = data[0]
            ids = data[1]
            mask = data[2]
            # http_receiver.pop_incoming_queue()
            start_time = time.time()

            for k in range(start_idx, 33):
                k = k - start_idx
                start_time_sub = time.time()
                print(next(models[k].parameters()).device)
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

            outgoing_queue.put(lm_logits)
            print('data store!!')

# Placeholder function for computation
def compute(data):
    # Replace this with your actual computation logic
    # For demonstration, let's just return the length of the data
    return len(data)



if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)



    start_idx = 5
    end_idx = 34
    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)


    #model = get_llm(args.ckpt_dir_hf, 'llm_weights')
    #tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    data_processor_thread = threading.Thread(target=process_data, args=[models, start_idx, end_idx, device])
    data_processor_thread.start()

    # Create the HTTP server process
    http_server_process = HTTPServer(('', args.server_port), HTTPRequestHandler)

    print('HTTP Server running on port 8080')

    try:
        http_server_process.serve_forever()
    except KeyboardInterrupt:
        http_server_process.server_close()
        data_processor_thread.terminate()

    # Wait for the data processor thread to finish (this won't happen in this example)
    data_processor_thread.join()
