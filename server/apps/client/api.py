'''
Client API Module
'''
import os
from helpers.http import get
from dotenv import load_dotenv
from helpers.logging import logger

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


def delete_job_at_middlewares(client_id: str, job_id: str):
    '''
    Delete Job at Middlware Job Registry
    '''

    clients = get_clients()

    for client_id_, client in clients.items():
        if client_id_ == client_id:
            get(f'http://{client["ip_address"]}:{client["http_port"]}/job_manager/delete', {
                'job_id': job_id})

            logger.info(f'Deleting Job at Middlware {client["id"]}')
