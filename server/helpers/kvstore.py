'''
Key Value Store Module
'''
import json
import os
import traceback
from typing import Any
from time import sleep
import redis
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

load_dotenv()

KVS_URL = os.getenv('KVSTORE_URL')
DELAY = float(os.getenv('DELAY'))

kvstore = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def kv_get(key: str) -> Any:
    '''
    Get Value from Key
    '''
    while True:
        try:
            reply = kvstore.get(key)
            break
        except Exception:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s.\n{traceback.format_exc()}')
            sleep(DELAY*6)

    if reply is None:
        return None

    return json.loads(reply)

def kv_set(key: str, value: Any) -> None:
    '''
    Set Value with Key
    '''
    while True:
        try:
            reply = kvstore.set(key, json.dumps(value))

            if not reply:
                raise Exception(f'Failed to Set key [{key}]!') 
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
            reply = kvstore.getdel(key)
            break
        except Exception:
            logger.error(
                f'KVStore Database Connection Error! Retrying in 30s.\n{traceback.format_exc()}')
            sleep(DELAY*6)

    if reply is None:
        return None

    return json.loads(reply)
