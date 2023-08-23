import os
from time import sleep
from dotenv import load_dotenv

from helpers.client_registration import register_client
from helpers.logging import logger


# import environment variables
load_dotenv()

# fetch mme server url
CLIENT_STATE = {
    'server_url': os.getenv('MANAGEMENT_SERVER')
}


# global variables
JOBS = {
    'job_ids': set(),
    'jobs': dict()
}

# register client machine on mme server
while True:  # loop through and wait for the server to be available for registration
    try:
        CLIENT_STATE['reginfo'] = register_client(CLIENT_STATE['server_url'])
        logger.info(f"registered client with {CLIENT_STATE['reginfo']}")
        break
    except:
        logger.warning(f'Failed to register client.')
        sleep(60)

# TODO: start process to keep alive and check for jobs
# TODO: for job found, will start a process to train and all
