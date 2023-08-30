'''
Client Status Management Module
'''
import os
from time import sleep
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

# import environment variables
load_dotenv()

DELAY = int(os.getenv('DELAY'))


def update_client_status(client_id: str, job_id: str, status: int, server_url: str):
    '''
    Update client's status to server.
    '''

    while True:
        # try to update the status
        logger.info(
            f'Updating client [{client_id}] status to [{status}] for Job [{job_id}]')
        post(f'{server_url}/job_manager/update_client_status', {
            'client_id': client_id,
            'job_id': job_id,
            'client_status': status
        })

        # wait for DELAY seconds
        sleep(DELAY)

        # get the updated status
        client_status = listen_to_client_status(client_id, job_id, server_url)

        if client_status == status:
            logger.info('Successfully updated client status!')
            break
        else:
            logger.info(
                f'Got Status [{client_status}], Expected [{status}] for Job [{job_id}]. Retrying...')


def listen_to_client_status(client_id: str, job_id: str, server_url: str):
    '''
    Method to listen to client status
    '''

    try:
        # listen to check if dataset flag is true or false
        url = f'{server_url}/job_manager/get'

        manifest = get(url, {'job_id': job_id})

        for client in manifest['exec_params']['client_info']:
            if client['client_id'] == client_id:
                return client['status']

    except Exception as e:
        logger.error(f'Failed to Client Stage. {e}')

    return -1
