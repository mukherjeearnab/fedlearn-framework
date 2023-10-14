'''
Client API Module
'''
from helpers.http import get
from global_kvset import app_globals


def get_clients():
    '''
    Returns a dict of all registered clients
    '''

    return get(f'{app_globals.get("LOOPBACK_URL")}/client_manager/get', {})


def get_alive_clients():
    '''
    Returns a dict of all alive clients
    '''

    return get(f'{app_globals.get("LOOPBACK_URL")}/client_manager/get_alive', {})
