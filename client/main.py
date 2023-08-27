'''
Client Main Module
'''
import os
from time import sleep
from dotenv import load_dotenv

from helpers.logging import logger
from apps.client.keep_alive import keep_alive_process
from apps.client.registration import register_client


# import environment variables
load_dotenv()

DELAY = int(os.getenv('DELAY'))

# fetch mme server url
CLIENT_STATE = {
    'server_url': os.getenv('MANAGEMENT_SERVER')
}


# global variables
JOBS = {
    'job_ids': [],
    'jobs': {}
}

# register client machine on mme server
while True:  # loop through and wait for the server to be available for registration
    try:
        CLIENT_STATE['reginfo'] = register_client(CLIENT_STATE['server_url'])
        logger.info(f"Registered client with {CLIENT_STATE['reginfo']}")
        break
    except Exception as e:
        logger.warning(f'Failed to register client. {e}')
        sleep(DELAY*12)

keep_alive_process(JOBS, CLIENT_STATE)
