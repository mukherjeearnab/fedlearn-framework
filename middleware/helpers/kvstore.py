'''
Key Value Store Module
'''
import json
import os
from typing import Any
from time import sleep
from global_kvset import app_globals
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

load_dotenv()

KVS_URL = os.getenv('KVSTORE_URL')
DELAY = float(os.getenv('DELAY'))


def kv_get(key: str) -> Any:
    '''
    Get Value from Key
    '''
    while True:
        try:
            reply = get(f'{KVS_URL}/get', {'key': key,
                        'client': f'middleware-{app_globals.get("HTTP_SERVER_PORT")}'})
            break
        except Exception as e:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s. {e}')
            sleep(DELAY*6)

    if reply['res'] == 404:
        return None

    return json.loads(reply['value'])


def kv_set(key: str, value: Any) -> None:
    '''
    Set Value with Key
    '''
    while True:
        try:
            post(f'{KVS_URL}/set', {'key': key, 'value': json.dumps(value),
                 'client': f'middleware-{app_globals.get("HTTP_SERVER_PORT")}'})
            break
        except Exception as e:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s. {e}')
            sleep(DELAY*6)


def kv_delete(key: str) -> Any:
    '''
    Delete Value with Key
    '''
    while True:
        try:
            reply = get(f'{KVS_URL}/delete', {'key': key,
                        'client': f'middleware-{app_globals.get("HTTP_SERVER_PORT")}'})
            break
        except Exception as e:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s. {e}')
            sleep(DELAY*6)

    if reply['res'] == 404:
        return None

    return reply['value']
