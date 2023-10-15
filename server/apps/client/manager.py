'''
Application Logic for Client Management
'''
from time import time
from helpers.kvstore import kv_get, kv_set
from helpers.semaphore import Semaphore


CLIENT_MAN_KEY = 'client_man_db'
UPDATE_LOCK = Semaphore()

# Client alove threshold in seconds
ALIVE_THRESHOLD = 10


class ClientManager:
    '''
    Client Count manager class
    '''

    def __init__(self):
        '''
        constructor
        '''
        # self.client_count
        # self.clients_info

        self._read_state()

    def _read_state(self):
        payload = kv_get(CLIENT_MAN_KEY)

        if payload is None:
            self.client_count = 0
            self.clients_info = {}

            self._update_state()
        else:
            self.client_count = payload['client_count']
            self.clients_info = payload['clients_info']

    def _update_state(self):
        data = {
            'clients_info': self.clients_info,
            'client_count': self.client_count,
        }
        kv_set(CLIENT_MAN_KEY, data)

    def get_clients(self) -> list:
        '''
        Get the dictionary of all registered clients.
        '''
        UPDATE_LOCK.wait()
        self._read_state()

        return self.clients_info

    def get_alive_clients(self) -> list:
        '''
        Returns a list of all alive clients, i.e., all clients
        who pinged atleast in the last 20 seconds 
        '''
        UPDATE_LOCK.wait()
        self._read_state()

        alive_clients = []

        for client_id in self.clients_info.keys():
            if (int(time()) - self.clients_info[client_id]['last_ping']) <= ALIVE_THRESHOLD:
                alive_clients.append(self.clients_info[client_id])

        return alive_clients

    def register_client(self, client_info: dict, ip_address: str) -> str:
        '''
        register function, increments the counter and returns id
        '''
        UPDATE_LOCK.acquire()
        self._read_state()

        self.client_count += 1

        if 'is_middleware' in client_info and client_info['is_middleware']:
            client_id = f'middleware-{self.client_count}'
            is_middleware = True
            http_port = client_info['http_port']
        else:
            client_id = f'client-{self.client_count}'
            is_middleware = False
            http_port = -1

        client = {
            'id': client_id,
            'hostname': client_info['sysinfo']['hostname'],
            'ip_address': ip_address,
            'last_ping': int(time()),
            'is_middleware': is_middleware,
            'http_port': http_port
        }

        self.clients_info[client_id] = client

        self._update_state()
        UPDATE_LOCK.release()

        return client['id']

    def alive_ping(self, client_id: str) -> bool:
        '''
        register function, increments the counter and returns id
        '''
        exec_resp = True
        UPDATE_LOCK.acquire()
        self._read_state()

        if client_id in self.clients_info:
            self.clients_info[client_id]['last_ping'] = int(time())
        else:
            exec_resp = False

        self._update_state()
        UPDATE_LOCK.release()

        return exec_resp
