'''
Client Main Module
'''
import os
from time import sleep
import argparse
from dotenv import load_dotenv
import prompt
from helpers import torch as _
from helpers.logging import logger
from apps.client.keep_alive import keep_alive_process
from apps.client.registration import register_client


# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--server")
parser.add_argument("-d", "--default", action="store_true")
args = parser.parse_args()


# fetch mme server url
CLIENT_STATE = {
    'server_url': os.getenv('MANAGEMENT_SERVER')
}

if args.default and args.server:
    print("Cannot use both --default and --server. Exiting...")
    exit()
if (not args.default) and (not args.server):
    prompt.management_server_prompt(CLIENT_STATE)
if args.server:
    prompt.override_server_url(args.server, CLIENT_STATE)

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
