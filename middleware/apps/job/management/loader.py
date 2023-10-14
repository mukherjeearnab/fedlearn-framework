'''
The Job Loader Module
'''

import os
# import json
from dotenv import load_dotenv
from helpers.logging import logger
# from apps.job.management.dataset import prepare_dataset_for_deployment
from apps.client.api import get_alive_clients

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def load_job(job_name: str, config: dict, config_registry: dict):
    '''
    Load the Job Config and perform some preliminary validation.
    '''
    logger.info(f'Loaded Job Config: \n{job_name}')

    # # prepare dataset for clients to download
    # logger.info(f'Starting Dataset Preperation for Job {job_name}')
    # prepare_dataset_for_deployment(config)
    # logger.info(f'Dataset Preperation Complete for Job {job_name}')

    # get available clients from the server registry
    client_list = get_alive_clients()

    config_registry[job_name] = config

    # check if sufficient clients are available or not, else return
    if len(client_list) < config['client_params']['num_clients']:
        logger.error(f'''Not Enough Clients available on register. Please register more clients.
                     Required: {config['client_params']['num_clients']} Available: {len(client_list)}''')

        return False

    print('Job Config Loaded.')

    return True
