import http.client
import os.path
import pickle
import msgpack
import lz4.frame
import argparse
import time

import torch
import yaml
import gc
from queue import Queue

import http_receiver

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
parser.add_argument('--selection', type=int)
parser.add_argument('--head', type=int)
args = parser.parse_args()

'''text = 'fodge'
newx = pickle.dumps(text)
total_size = len(newx)

#conn = http.client.HTTPConnection('10.100.218.157', 80)
conn = http.client.HTTPConnection('test-service.nrp-nautilus.io')
conn.connect()


conn.putrequest('POST', '/upload/')
conn.putheader('Content-Type', 'application/octet-stream')
conn.putheader('Content-Length', str(total_size))
conn.endheaders()


print(newx)
conn.send(newx)
resp = conn.getresponse()'''

#returning_queue = []
returning_queue = Queue()

def get_queue_data():
    '''if len(returning_queue) > 0:
        return returning_queue[0]
    else:
        return []'''
    #while returning_queue.empty():
    #    time.sleep(0.5)
    data = []
    while not returning_queue.empty():
        data.append(returning_queue.get())

    return data


def pop_incoming_queue():
    returning_queue.get()


def send_data(server_ip, server_port, text, calculate_opt, timestamp_manager):
    start_time = time.time()
    '''start_idx = text[0]
    idx = text[4]
    input = text[1]
    client_comp_time = text[5]'''

    #newx = pickle.dumps(text)
    #print('text: ',text)
    #packed_data = msgpack.packb(text, use_bin_type=True)
    #compressed_data = lz4.frame.compress(packed_data)

    total_size = len(text)
    print('communication size: ', total_size)

    #start_time = time.time()
    conn = http.client.HTTPConnection(server_ip, server_port)
    conn.connect()

    #conn.putrequest('POST', '/upload/')
    conn.putrequest('POST', '/')
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Content-Length', str(total_size))
    conn.endheaders()

    conn.send(text)
    end_time = time.time()

    start_time2 = time.time()
    resp = conn.getresponse()

    resp_data = resp.readlines()
    resp_str = b''

    for i in range(4, len(resp_data)):
        resp_str = resp_str + resp_data[i]
    end_time2 = time.time()
    rtt = end_time2 - start_time

    try:
        # resp_message = [start_idx, total_comp_time, idx]
        resp_message = pickle.loads(resp_str)

        resp_message = resp_message[0]
        print('server side resp: ', resp_message)

        resp_message.append(rtt)    #resp_message = [start_idx, total_comp_time, idx, rtt(total time)]

        if not resp_message[0] == -1:
            timestamp_manager.end_times = (resp_message[2], end_time2)

        if not (resp_message[0] == 0 or resp_message[0] == -1):
            print('data stored!')
            calculate_opt.incoming_count = calculate_opt.incoming_count + 1
            calculate_opt.server_comp_statistics = (resp_message[0], resp_message[3])
    except:
        print('error')
    #print('http receiving: ', start_idx, rtt)
    print('rrt: ', rtt)
    gc.collect()

    #middle devices used only
    #if client_comp_time is not None:
    #    http_receiver.outgoing_queue.put([start_idx, rtt + client_comp_time, idx])




if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    text = 'fodge'
    send_data(args.server_ip, args.server_port, text)


