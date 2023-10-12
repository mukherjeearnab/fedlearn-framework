'''
Client API Module
'''
from helpers.http import get
from global_kvset import app_globals

SERVER_URL = f'http://localhost:{app_globals.get("SERVER_PORT")}'

print(f'Loaded HTTP Server URL: {SERVER_URL}')


def get_clients():
    '''
    Returns a dict of all registered clients
    '''

    return get(f'{SERVER_URL}/client_manager/get', {})


def get_alive_clients():
    '''
    Returns a dict of all alive clients
    '''

    return get(f'{SERVER_URL}/client_manager/get_alive', {})
