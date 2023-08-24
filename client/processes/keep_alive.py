'''
The Keep Alive Process Module
Here we host the methods to keep alive the client,
and periodically checking the server stats for jobs.
'''
from time import sleep
from helpers.http import post
from helpers.logging import logger
from processes.job import get_jobs_from_server


def keep_alive_process(jobs_registry: dict, client_state: dict):
    '''
    The Keep Alive Process Method
    '''
    logger.info('Starting Keep Alive Process.')

    # the infinite loop
    while True:

        # keep alive logic
        try:
            # ping the server
            send_keep_alive(client_state['reginfo']['id'],
                            client_state['server_url'])
        except:
            logger.warning('Failed to ping server for keep alive function.')

        # job check logic
        try:
            get_jobs_from_server(client_state['reginfo']['id'],
                                 jobs_registry, client_state['server_url'])
        except:
            logger.warning('Failed to ping server for keep alive function.')

        sleep(5)


def send_keep_alive(client_id: str, server_url: str):
    '''
    Keep Alive HTTP request sender method
    '''

    url = f'{server_url}/client_manager/ping'

    body = {
        'client_id': client_id
    }

    logger.info(f'Sending client ping request to Server at {url}')

    post(url, body)
