'''
Middleware Registration Module
'''

from helpers.http import post
from helpers.configuration import get_system_info
from helpers.logging import logger
from global_kvset import app_globals


def register_middleware(mme_server_url: str) -> dict:
    '''
    Middleware Registration Method
    '''
    url = f'{mme_server_url}/client_manager/register'

    body = {
        'sysinfo': get_system_info(),
        'is_middleware': True,
        'http_port': app_globals.get("HTTP_SERVER_PORT")
    }

    logger.info(
        f'Sending middleware registration info {body} request to {url}')

    middleware_regid = post(url, body)

    logger.info(f'Received middleware registration {middleware_regid}')

    return middleware_regid
