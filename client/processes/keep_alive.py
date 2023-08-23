'''
The Keep Alive Process Module
Here we host the methods to keep alive the client,
and periodically checking the server stats for jobs.
'''
from time import sleep
from helpers.http import get, post
from helpers.logging import logger


def keep_alive_process(jobs_register: dict, client_state: dict):
    '''
    The Keep Alive Process Method
    '''
    logger.info('Starting Keep Alive Process.')

    while True:
        try:
            send_keep_alive(client_state['reginfo']['id'],
                            client_state['server_url'])
        except Exception as e:
            logger.warning(f'Failed to register client. {e.args}')
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
