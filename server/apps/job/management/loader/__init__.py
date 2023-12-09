'''
The Job Loader Module
'''

import os
import json
import argparse
from dotenv import load_dotenv
from helpers.logging import logger
from helpers.file import read_yaml_file
from apps.job.management.dataset import prepare_dataset_for_deployment
from apps.job.management.loader import single_level, hierarchical
from apps.client.api import get_alive_clients

# import environment variables
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port")
args = parser.parse_args()

SERVER_PORT = int(args.port if args.port else os.getenv('SERVER_PORT'))
DELAY = float(os.getenv('DELAY'))


def load_job(job_name: str, job_config: str, config_registry: dict):
    '''
    Load the Job Config and perform some preliminary validation.
    '''
    # load the job config file
    config = read_yaml_file(f'../templates/job_config/{job_config}.yaml')

    logger.info(f'Reading Job Config: \n{job_config}')

    config_registry[job_name] = config

    print('Job Config: ', json.dumps(config, sort_keys=True, indent=4))

    # load the python module files for the configuration
    if 'hierarchical' in config and config['hierarchical']:
        config = hierarchical.load_module_files(config)
        num_clients = len(config['middleware_params']['individual_configs'])
    else:
        config = single_level.load_module_files(config)
        num_clients = config['client_params']['num_clients']

    logger.info('Loaded Py Modules from Job Config')

    # prepare dataset for clients to download
    logger.info(f'Starting Dataset Preperation for Job {job_name}')
    prepare_dataset_for_deployment(config)
    logger.info(f'Dataset Preperation Complete for Job {job_name}')

    # get available clients from the server registry
    client_list = get_alive_clients()

    # check if sufficient clients are available or not, else return
    if len(client_list) < num_clients:
        logger.warning(f'''Not Enough Clients available on register. Please register more clients.
                     Required: {num_clients} Available: {len(client_list)}''')

    print('Job Config Loaded.')
