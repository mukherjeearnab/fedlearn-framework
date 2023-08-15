import os
from time import sleep
from dotenv import load_dotenv

from helpers.client_registration import register_client
from helpers.logging import logger


# import environment variables
load_dotenv()

# fetch mme server url
MANAGEMENT_SERVER = os.getenv('MANAGEMENT_SERVER')

# register client machine on mme server
while True:  # loop through and wait for the server to be available for registration
    try:
        CLIENT_REGINFO = register_client(MANAGEMENT_SERVER)
        logger.info(f'registered client with {CLIENT_REGINFO}')
        break
    except Exception as e:
        logger.warning(f'Failed to register client. {e.args}')
        sleep(60)
