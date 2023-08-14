'''
HTTP comms module, with simplified syntax to avoid code repetition
'''
import requests


def get(url: str, params: dict) -> dict:
    '''
    GET request method
    '''
    req = requests.get(url, params, timeout=10)
    data = req.json()

    return data


def post(url: str, params: dict) -> dict:
    '''
    POST request method
    '''
    req = requests.post(url, json=params, timeout=10)
    data = req.json()

    return data
