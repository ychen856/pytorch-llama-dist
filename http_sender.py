import http.client
import os.path
import pickle
import argparse
import yaml


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

def send_data(server_address, text):
    newx = pickle.dumps(text)
    total_size = len(newx)

    conn = http.client.HTTPConnection(server_address)
    conn.connect()

    conn.putrequest('POST', '/upload/')
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Content-Length', str(total_size))
    conn.endheaders()

    print(text)
    print(newx)
    conn.send(newx)
    resp = conn.getresponse()

    print('resp: ', resp)



if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    text = 'fodge'
    send_data(args.server_address, text)






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
