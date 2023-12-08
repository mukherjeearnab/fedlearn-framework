'''
This is the Commandline interface for managing the server
'''
import os
import logging
from time import sleep
import traceback
from dotenv import load_dotenv
import prompt
from helpers.logging import logger
from apps.http_server.controller import start_server
from apps.middleware.keep_alive import keep_alive_process
from apps.middleware.registration import register_middleware


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


SINGLE_COMMANDS = ['exit']

# import environment variables
load_dotenv()


DELAY = float(os.getenv('DELAY'))

# fetch mme server url
CLIENT_STATE = {
    'server_url': os.getenv('MANAGEMENT_SERVER')
}

prompt.management_server_prompt(CLIENT_STATE)

prompt.http_server_prompt()


# global variables
JOBS = {
    'job_ids': [],
    'jobs': {}
}

# 0. Start the Middlware HTTP Server
start_server()

# 1. register middleware on server
while True:  # loop through and wait for the server to be available for registration
    try:
        CLIENT_STATE['reginfo'] = register_middleware(
            CLIENT_STATE['server_url'])
        logger.info(f"Registered client with {CLIENT_STATE['reginfo']}")
        break
    except Exception:
        logger.warning(f'Failed to register client.\n{traceback.format_exc()}')
        sleep(DELAY*12)

# 2. Start KeepAlive Process to listen to jobs from server
keep_alive_process(JOBS, CLIENT_STATE)
