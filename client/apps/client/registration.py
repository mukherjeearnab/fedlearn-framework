'''
Client Registration Module
'''

from helpers.http import post
from helpers.configuration import get_system_info
from helpers.logging import logger


def register_client(mme_server_url: str) -> dict:
    '''
    Client Registration Method
    '''
    url = f'{mme_server_url}/client_manager/register'

    body = {
        'sysinfo': get_system_info()
    }

    logger.info(f'Sending client registration info {body} request to {url}')

    client_regid = post(url, body)

    logger.info(f'Received client registration {client_regid}')

    return client_regid
