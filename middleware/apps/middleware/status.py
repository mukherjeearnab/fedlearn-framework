'''
middleware Status Management Module
'''
import os
from time import sleep
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def update_middleware_status(middleware_id: str, job_id: str, status: int, server_url: str):
    '''
    Update middleware's status to server.
    '''

    while True:
        # try to update the status
        logger.info(
            f'Updating middleware [{middleware_id}] status to [{status}] for Job [{job_id}]')
        post(f'{server_url}/job_manager/update_client_status', {
            'client_id': middleware_id,
            'job_id': job_id,
            'client_status': status
        })

        # wait for DELAY seconds
        sleep(DELAY)

        # get the updated status
        middleware_status = listen_to_middleware_status(
            middleware_id, job_id, server_url)

        if middleware_status == status:
            logger.info('Successfully updated middleware status!')
            break
        else:
            logger.info(
                f'Got Status [{middleware_status}], Expected [{status}] for Job [{job_id}]. Retrying...')


def listen_to_middleware_status(middleware_id: str, job_id: str, server_url: str):
    '''
    Method to listen to middleware status
    '''

    try:
        # listen to check if dataset flag is true or false
        url = f'{server_url}/job_manager/get_exec'

        manifest = get(url, {'job_id': job_id})

        for middleware in manifest['job_status']['client_info']:
            if middleware['client_id'] == middleware_id:
                return middleware['status']

    except Exception as e:
        logger.error(f'Failed to middleware Stage. {e}')

    return -1
