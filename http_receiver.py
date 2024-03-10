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
        print(post_data)
        print(pickle.loads(post_data))

        # Process the received data here:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Data received successfully!')
        #body = self.get_body(conn, size)
        #data = pickle.loads(post_data)
        #print(data)

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

    #sending response
    text = 'this is response!'
    http_sender.send_data(args.client_data, text)
    #server_class = HTTPServer
    #run(server_class, 5, args.server_port)

'''if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()'''