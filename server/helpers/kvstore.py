'''
Key Value Store Module
'''
import json
import os
import traceback
from typing import Any
from time import sleep
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
            reply = get(f'{KVS_URL}/get', {'key': key, 'client': 'server'})
            break
        except Exception:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s.\n{traceback.format_exc()}')
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
                                    'client': 'server'})
            break
        except Exception:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s.\n{traceback.format_exc()}')
            sleep(DELAY*6)


def kv_delete(key: str) -> Any:
    '''
    Delete Value with Key
    '''
    while True:
        try:
            reply = get(f'{KVS_URL}/delete', {'key': key, 'client': 'server'})
            break
        except Exception:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s.\n{traceback.format_exc()}')
            sleep(DELAY*6)

    if reply['res'] == 404:
        return None

    return reply['value']
