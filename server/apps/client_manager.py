'''
Application Logic for Client Management
'''
from helpers.kvstore import kv_get, kv_set
CLIENT_MAN_KEY = 'client_man_db'


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
        # self.reg_lock

        self._read_state()

    def _read_state(self):
        payload = kv_get(CLIENT_MAN_KEY)

        if payload is None:
            self.client_count = 0
            self.clients_info = []
            self.reg_lock = False

            self._update_state()
        else:
            self.client_count = payload['client_count']
            self.clients_info = payload['clients_info']
            self.reg_lock = payload['reg_lock']

    def _update_state(self):
        data = {
            'clients_info': self.clients_info,
            'client_count': self.client_count,
            'reg_lock': self.reg_lock
        }
        kv_set(CLIENT_MAN_KEY, data)

    def show_clients(self,) -> None:
        '''
        Print all the clients and their details
        '''
        self._read_state()

        print(self.clients_info)

    def register_client(self, client_info: dict, ip_address: str) -> str:
        '''
        register function, increments the counter and returns id
        '''
        self._read_state()

        if self.reg_lock:
            return None

        self.client_count += 1

        client = {
            'name': f'client-{self.client_count}',
            'hostname': client_info['sysinfo']['hostname'],
            'ip_address': ip_address
        }

        self.clients_info.append(client)

        self._update_state()

        return client['name']
