#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import pickle
import argparse
import yaml
import http_sender

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

incoming_queue = []
outgoing_queue = []
def get_queue_data():
    if len(incoming_queue) > 0:
        return incoming_queue[0]
    else:
        return []

def set_outgoing_queue(outputs):
    outgoing_queue.append(outputs)

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        print('receive POST:')
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self._set_headers()
        print('length: ', content_length)
        #print(post_data)

        decrypt_data = pickle.loads(post_data)
        print(decrypt_data)

        incoming_queue.append(decrypt_data)

        # Process the received data here:
        '''self.send_response(200)
        self.end_headers()

        newx = pickle.dumps('Data received successfully!')
        self.wfile.write(newx)'''


        self.return_message()

    def return_message(self):
        while 1:
            if len(outgoing_queue) > 0:
                break
        # Process the received data here:
        self.send_response(200)
        self.end_headers()

        newx = pickle.dumps('Data received successfully!')
        self.wfile.write(newx)

def run(server_class=HTTPServer, handler_class=S, server_ip='', port=80):
    #server_address = ('localhost', port)
    server_address = (server_ip, port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('port: ', args.server_port)
    run(port=args.server_port)



'''if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()'''