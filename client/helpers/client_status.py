'''
Client Status Management Module
'''
from helpers.http import post
from helpers.logging import logger


def update_client_status(client_id: str, job_id: str, status: int, server_url: str):
    '''
    Update client's status to server.
    '''

    logger.info(f'Updating client status to [{status}] for Job [{job_id}]')
    post(f'{server_url}/job_manager/update_client_status', {
        'client_id': client_id,
        'job_id': job_id,
        'client_status': status
    })
