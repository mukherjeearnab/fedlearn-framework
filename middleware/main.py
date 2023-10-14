'''
This is the Commandline interface for managing the server
'''
import os
import logging
from time import sleep
from dotenv import load_dotenv
from global_kvset import app_globals
from helpers.logging import logger
from apps.http_server.controller import start_server
from apps.middleware.keep_alive import keep_alive_process
from apps.middleware.registration import register_middleware


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


SINGLE_COMMANDS = ['exit']

# import environment variables
load_dotenv()

SERVER_PORT = int(input('Enter HTTP server port: ').strip())
app_globals.set('LOOPBACK_URL', f'http://localhost:{SERVER_PORT}')
app_globals.set('HTTP_SERVER_PORT', SERVER_PORT)

DELAY = float(os.getenv('DELAY'))

# fetch mme server url
CLIENT_STATE = {
    'server_url': os.getenv('MANAGEMENT_SERVER')
}


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
    except Exception as e:
        logger.warning(f'Failed to register client. {e}')
        sleep(DELAY*12)

# 2. Start KeepAlive Process to listen to jobs from server
keep_alive_process(JOBS, CLIENT_STATE)
