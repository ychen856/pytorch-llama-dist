import http.client
import os.path
import pickle
import argparse
import yaml
import http_receiver

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

returning_queue = []

def get_queue_data():
    if len(returning_queue) > 0:
        return returning_queue[0]
    else:
        return []

def send_data(server_ip, server_port, text):
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
    resp = conn.getresponse()

    resp_message =  pickle.loads(resp.readlines()[4])
    returning_queue.append(resp_message)
    print('resp: ', resp_message)



if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    text = 'fodge'
    send_data(args.server_ip, args.server_port, text)


    # receiveing response
    http_receiver.run(server_ip=args.client_ip, port=args.client_port)







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
