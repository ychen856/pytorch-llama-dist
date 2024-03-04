#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import pickle

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
        #body = self.get_body(conn, size)
        #data = pickle.loads(post_data)
        #print(data)

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()