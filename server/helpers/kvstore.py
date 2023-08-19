from helpers.http import get, post
import json
import os
from typing import Any
from time import sleep
from dotenv import load_dotenv
from helpers.logging import logger

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
        except:
            logger.warning(
                'KVStore Database Connection Error! Retrying in 30s.')
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
        except:
            logger.warning(
                'KVStore Database Connection Error! Retrying in 30s.')
            sleep(30)