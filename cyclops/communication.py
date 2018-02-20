import os
from threading import Thread
import socket
import ast
import time

from signal import signal, SIGPIPE, SIG_DFL
from cyclops.type_hints import port_type, Optional, Dict


signal(SIGPIPE,SIG_DFL)

class PoseServer(Thread):

    def __init__(self, host: Optional[str] = None, 
                 port: Optional[port_type] = 12345):
        Thread.__init__(self)
        self.port = port
        self.socket = socket.socket()
        self.host = socket.gethostname() if not host else host
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen()

        self.belief = {'x': 0., 'y': 0., 'theta': 0.}
        self.is_running = True
        self.start()
    
    def run(self):
        while self.is_running:
            client_socket, address = self.socket.accept()
            data = client_socket.recv(1024)
            data = ast.literal_eval(data.decode())

            if isinstance(data, dict) and self.is_correct_format(data):
                if self.is_update_data(data):
                    self.handle_update(data)
                    self.handle_query(client_socket)

                elif self.is_regular_query(data):
                    self.handle_query(client_socket)

            client_socket.close()    
        self.socket.close()            
    
    def is_correct_format(self, data: Dict):
        return all([field in data for field in ['purpose', 'id', 'timestamp']])
    
    def is_update_data(self, data: Dict):
        return data['purpose'] == 'update'
    
    def is_regular_query(self, data: Dict):
        return data['purpose'] == 'get_pose'
    
    def handle_update(self, data: Dict):
        try:
            for key in ['x', 'y', 'theta']:
                self.belief[key] = data[key]
        except Exception as ex:
            print(ex)
    
    def handle_query(self, client_socket: socket.socket):
        client_socket.send(str.encode(str(self.belief)))

    def stop(self):
        self.is_running = False        


class PoseClient(object):

    def __init__(self, host: Optional[str] = None, 
                 port: Optional[port_type] = 12345):
        self.port = port
        self.host = socket.gethostname() if not host else host
        self.server_address = (self.host, self.port)
        self.is_running = True

    def run(self):
        while self.is_running:
            print(self.query())
            time.sleep(0.1)
        
    def query(self):
        _query_bytes = str.encode(str({'purpose': 'get_pose', 
                                       'id': 'pose_client',
                                       'timestamp': time.time()}))
        self.socket = socket.socket()
        self.socket.connect(self.server_address)   
        self.socket.send(_query_bytes)
        data = self.socket.recv(1024).decode()
        self.socket.close()
        return data

    def stop(self):
        self.is_running = False


if __name__ == '__main__':
    client = PoseClient(port=42420)
    client.run()
