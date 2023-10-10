'''
The Keep Alive Process Module
Here we host the methods to keep alive the client,
and periodically checking the server stats for jobs.
'''
from time import sleep
import os
from helpers.http import post
from helpers.logging import logger
from apps.job.getters import get_jobs_from_server
from dotenv import load_dotenv

# import environment variables
load_dotenv()
DELAY = float(os.getenv('DELAY'))


def keep_alive_process(jobs_registry: dict, middleware_state: dict):
    '''
    The Keep Alive Process Method
    '''
    logger.info('Starting Keep Alive Process.')

    # the infinite loop
    while True:

        # keep alive logic
        try:
            # ping the server
            send_keep_alive(middleware_state['reginfo']['id'],
                            middleware_state['server_url'])
        except:
            # logger.warning('Failed to ping server for keep alive function.')
            pass

        # job check logic
        try:
            get_jobs_from_server(middleware_state['reginfo']['id'],
                                 jobs_registry, middleware_state['server_url'])
        except:
            # logger.warning('Failed to ping server for keep alive function.')
            pass
        sleep(DELAY)


def send_keep_alive(client_id: str, server_url: str):
    '''
    Keep Alive HTTP request sender method
    '''

    url = f'{server_url}/client_manager/ping'

    body = {
        'client_id': client_id
    }

    # logger.info(f'Sending client ping request to Server at {url}')

    post(url, body)
