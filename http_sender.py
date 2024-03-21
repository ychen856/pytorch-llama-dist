import http.client
import os.path
import pickle
import argparse
import time

import yaml
import http_receiver

from queue import Queue

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
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
    while returning_queue.empty():
        time.sleep(0.5)

    return returning_queue.get()


def pop_incoming_queue():
    returning_queue.pop(0)

def send_data(server_ip, server_port, text):
    start_time = time.time()
    newx = pickle.dumps(text)
    total_size = len(newx)

    conn = http.client.HTTPConnection(server_ip, server_port)
    conn.connect()

    conn.putrequest('POST', '/upload/')
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Content-Length', str(total_size))
    conn.endheaders()

    print(text)
    #print(newx)
    conn.send(newx)
    end_time = time.time()
    print('client sending time: ', end_time - start_time)


    start_time2 = time.time()
    resp = conn.getresponse()

    resp_data = resp.readlines()
    resp_str = b''
    for i in range(5, len(resp_data)):
        resp_str = resp_str + resp_data[i]


    resp_message = pickle.loads(resp_str)
    #returning_queue.append(resp_message)
    returning_queue.put(resp_message)
    print('resp: ', resp_message)
    end_time2 = time.time()
    print('client receiving time: ', end_time2 - start_time2)
    print('rrt: ', end_time2 - start_time)



if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    text = 'fodge'
    send_data(args.server_ip, args.server_port, text)









'''while True:
    #chunk = newx[:1024]
    #newx = newx[1024:]
    #newx = pickle.dumps(newx)
    chunk = newx
    print('chunk: ', chunk)

    if not chunk:
        break
    conn.send(chunk)
resp = conn.getresponse()'''
