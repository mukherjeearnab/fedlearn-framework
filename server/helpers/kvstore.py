'''
Key Value Store Module
'''
import json
import os
from typing import Any
from time import sleep
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

load_dotenv()

KVS_URL = os.getenv('KVSTORE_URL')


def kv_get(key: str) -> Any:
    '''
    Get Value from Key
    '''
    while True:
        try:
            reply = get(f'{KVS_URL}/get', {'key': key})
            break
        except Exception as e:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s. {e}')
            sleep(30)

    if reply['res'] == 404:
        return None

    return json.loads(reply['value'])


def kv_set(key: str, value: Any) -> None:
    '''
    Set Value with Key
    '''
    while True:
        try:
            post(f'{KVS_URL}/set', {'key': key, 'value': json.dumps(value)})
            break
        except Exception as e:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s. {e}')
            sleep(30)
