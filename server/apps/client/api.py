'''
Client API Module
'''
import os
import argparse
from helpers.http import get
from dotenv import load_dotenv
from helpers.logging import logger

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port")
args = parser.parse_args()

SERVER_PORT = int(args.port if args.port else os.getenv('SERVER_PORT'))

SERVER_URL = f'http://localhost:{SERVER_PORT}'


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
