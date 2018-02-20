import unittest
import socket
import time
import ast

from cyclops.test.testing_utility import CyclopsUnitTestBase
from cyclops.communication import PoseServer


class PoseServerTests(CyclopsUnitTestBase):

    @classmethod
    def setUpClass(cls):
        cls.__host = 'localhost'
        cls.__port = 12345
        cls.__server = PoseServer(cls.__host, cls.__port)
        cls.__query_bytes = str.encode(str({
            'purpose': 'get_pose', 'id': 'robbie', 'timestamp': time.time()}))
        cls.__expected_data_after_query = {'x': 0.15, 'y': 0.2, 'theta': 1.57}
        cls.__data_to_update = str.encode(str({
            'purpose': 'update', 'id': 'member0', 'timestamp': time.time(),
            'x': 0.35, 'y': 0.5, 'theta': 2.0}))
        cls.__expected_belief_after_update = {'x': 0.35, 'y': 0.5, 'theta': 2.0}

    def setUp(self):
        self.__server.belief = {'x': 0.15, 'y': 0.2, 'theta': 1.57}
        self.__client_socket = socket.socket()

    @classmethod
    def tearDownClass(cls):
        cls.__server.stop()

    def test_query(self):
        self.__client_socket.connect((self.__host, self.__port))
        self.__client_socket.send(self.__query_bytes)
        _data = ast.literal_eval(self.__client_socket.recv(1024).decode())
        self.assertDictEqual(_data, self.__expected_data_after_query)

    def test_update(self):
        self.__client_socket.connect((self.__host, self.__port))
        self.__client_socket.send(self.__data_to_update)
        _data = ast.literal_eval(self.__client_socket.recv(1024).decode())
        self.assertDictEqual(_data, self.__expected_belief_after_update)


if __name__ == '__main__':
    unittest.main()
    