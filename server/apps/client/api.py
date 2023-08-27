'''
Client API Module
'''
import os
from helpers.http import get
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = f'http://localhost:{int(os.getenv("SERVER_PORT"))}'


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
