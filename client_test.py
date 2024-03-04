import http.client
import os.path
import pickle

#total_size = os.path.getsize('D:/workspace/pytorch-llama-dist/sep.log')
#infile = open('D:/workspace/pytorch-llama-dist/sep.log')

text = 'fodge'
newx = pickle.dumps(text)
total_size = len(newx)

conn = http.client.HTTPConnection('localhost', 80)
#conn = http.client.HTTPConnection('test-service.nrp-nautilus.io')
conn.connect()


conn.putrequest('POST', '/upload/')
conn.putheader('Content-Type', 'application/octet-stream')
conn.putheader('Content-Length', str(total_size))
conn.endheaders()


print(newx)
conn.send(newx)
resp = conn.getresponse()

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
