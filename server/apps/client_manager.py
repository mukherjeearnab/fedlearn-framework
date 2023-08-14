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
        self.__payload = kv_get(CLIENT_MAN_KEY)

        if self.__payload is None:
            self.client_count = 0
            self.clients_info = []
            kv_set(CLIENT_MAN_KEY, self.clients_info)
        else:
            self.client_count = len(self.__payload)
            self.clients_info = self.__payload

    def show_clients(self,) -> None:
        '''
        Print all the clients and their details
        '''
        self.__payload = kv_get(CLIENT_MAN_KEY)
        self.client_count = len(self.__payload)
        self.clients_info = self.__payload

        print(self.clients_info)

    def register_client(self, client_info: dict, ip_address: str) -> str:
        '''
        register function, increments the counter and returns id
        '''
        self.client_count += 1

        client = {
            'name': f'client-{self.client_count}',
            'hostname': client_info['sysinfo']['hostname'],
            'ip_address': ip_address
        }

        self.clients_info.append(client)
        kv_set(CLIENT_MAN_KEY, self.clients_info)

        return client['name']
