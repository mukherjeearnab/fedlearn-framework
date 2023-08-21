'''
Application Logic for Client Management
'''
from helpers.kvstore import kv_get, kv_set
from helpers.semaphore import Semaphore

CLIENT_MAN_KEY = 'client_man_db'
UPDATE_LOCK = Semaphore()


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
            self.clients_info = []

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

    def get_clients(self,) -> list:
        '''
        Print all the clients and their details
        '''
        UPDATE_LOCK.wait()
        self._read_state()

        return self.clients_info

    def register_client(self, client_info: dict, ip_address: str) -> str:
        '''
        register function, increments the counter and returns id
        '''
        UPDATE_LOCK.acquire()
        self._read_state()

        self.client_count += 1

        client = {
            'name': f'client-{self.client_count}',
            'hostname': client_info['sysinfo']['hostname'],
            'ip_address': ip_address
        }

        self.clients_info.append(client)

        self._update_state()
        UPDATE_LOCK.release()

        return client['name']
