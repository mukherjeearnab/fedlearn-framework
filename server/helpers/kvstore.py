'''
Key Value Store Module
'''
import json
import os
import traceback
from typing import Any
from time import sleep
import redis
from helpers.argsparse import args
from helpers.http import get, post
from helpers.logging import logger
from dotenv import load_dotenv

load_dotenv()

KVS_URL = os.getenv('KVSTORE_URL')
DELAY = float(os.getenv('DELAY'))

if args.redis:
    kvstore = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


#########################
# Wrapper KVStore API
#########################

def kv_get(key: str) -> Any:
    '''
    Get Value from Key
    '''
    if args.redis:
        reply = kv_get_redis(key)
    else:
        reply = kv_get_legacy(key)

    return reply


def kv_set(key: str, value: Any) -> None:
    '''
    Set Value with Key
    '''
    if args.redis:
        reply = kv_set_redis(key, value)
    else:
        reply = kv_set_legacy(key, value)

    return reply


def kv_delete(key: str) -> Any:
    '''
    Delete Value with Key
    '''
    if args.redis:
        reply = kv_delete_redis(key)
    else:
        reply = kv_delete_legacy(key)

    return reply


#########################
# Redis-based KVStore API
#########################

def kv_get_redis(key: str) -> Any:
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

def kv_set_redis(key: str, value: Any) -> None:
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


def kv_delete_redis(key: str) -> Any:
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


###############################
# HTTP-based Legacy KVStore API
###############################

def kv_get_legacy(key: str) -> Any:
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


def kv_set_legacy(key: str, value: Any) -> None:
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


def kv_delete_legacy(key: str) -> Any:
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
