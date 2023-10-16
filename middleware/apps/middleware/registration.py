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

    parent_prefix = middleware_regid['id'].split('-')[0]
    node_id = middleware_regid['id'].split('-')[-1]

    app_globals.set("MIDDLEWARE_ID", middleware_regid['id'])
    app_globals.set("PARENT_NODE_PREFIX", f'{parent_prefix}_mw{node_id}')
    app_globals.save()

    logger.info(
        f'Setting PARENT_NODE_PREFIX to [{app_globals.get("PARENT_NODE_PREFIX")}]')

    return middleware_regid
