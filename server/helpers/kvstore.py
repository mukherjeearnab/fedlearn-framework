from helpers.http import get, post
import json
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

KVS_URL = os.getenv('KVSTORE_URL')


def kv_get(key: str) -> Any:
    '''
    Get Value from Key
    '''
    reply = get(f'{KVS_URL}/get', {'key': key})

    if reply['res'] == 404:
        return None

    return json.loads(reply['value'])


def kv_set(key: str, value: Any) -> None:
    '''
    Set Value with Key
    '''
    post(f'{KVS_URL}/set', {'key': key, 'value': json.dumps(value)})
