'''
The Job Loader Module
'''

import os
import json
from dotenv import load_dotenv
from helpers.logging import logger
from helpers.file import read_yaml_file, read_py_module
from apps.job.management.dataset import prepare_dataset_for_deployment
from apps.client.api import get_alive_clients

# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
DELAY = int(os.getenv('DELAY'))


def load_job(job_name: str, config_registry: dict):
    '''
    Load the Job Config and perform some preliminary validation.
    '''
    # load the job config file
    config = read_yaml_file(f'./templates/job_config/{job_name}.yaml')

    logger.info(f'Reading Job Config: \n{job_name}')

    config_registry[job_name] = config

    print('Job Config: ', json.dumps(config, sort_keys=True, indent=4))

    # load the python module files for the configuration
    config = load_module_files(config)
    logger.info('Loaded Py Modules from Job Config')

    # prepare dataset for clients to download
    logger.info(f'Starting Dataset Preperation for Job {job_name}')
    prepare_dataset_for_deployment(config)
    logger.info(f'Dataset Preperation Complete for Job {job_name}')

    # get available clients from the server registry
    client_list = get_alive_clients()

    # check if sufficient clients are available or not, else return
    if len(client_list) < config['client_params']['num_clients']:
        logger.warning(f'''Not Enough Clients available on register. Please register more clients.
                     Required: {config['client_params']['num_clients']} Available: {len(client_list)}''')

    print('Job Config Loaded.')


def load_module_files(config: dict) -> dict:
    '''
    This Module Loads the configuration files (.py modules) 
    as strings and adds them to the configuration dictionary.
    '''

    # read client_params.dataset.preprocessor
    config['client_params']['dataset']['preprocessor'] = {
        'file': config['client_params']['dataset']['preprocessor'],
        'content': read_py_module(
            f"./templates/preprocess/{config['client_params']['dataset']['preprocessor']}")
    }

    # read client_params.dataset.distribution.distributor
    config['client_params']['dataset']['distribution']['distributor'] = {
        'file': config['client_params']['dataset']['distribution']['distributor'],
        'content': read_py_module(
            f"./templates/distribution/{config['client_params']['dataset']['distribution']['distributor']}")
    }

    # read model_params.model_file
    config['client_params']['model_params']['model_file'] = {
        'file': config['client_params']['model_params']['model_file'],
        'content': read_py_module(
            f"./templates/models/{config['client_params']['model_params']['model_file']}")
    }

    # read model_params.parameter_mixer
    config['client_params']['model_params']['parameter_mixer'] = {
        'file': config['client_params']['model_params']['parameter_mixer'],
        'content': read_py_module(
            f"./templates/param_mixer/{config['client_params']['model_params']['parameter_mixer']}")
    }

    # read model_params.training_loop_file
    config['client_params']['model_params']['training_loop_file'] = {
        'file': config['client_params']['model_params']['training_loop_file'],
        'content': read_py_module(
            f"./templates/training/{config['client_params']['model_params']['training_loop_file']}")
    }

    # read model_params.test_file
    config['client_params']['model_params']['test_file'] = {
        'file': config['client_params']['model_params']['test_file'],
        'content': read_py_module(
            f"./templates/testing/{config['client_params']['model_params']['test_file']}")
    }

    # read server_params.aggregator
    config['server_params']['aggregator'] = {
        'file': config['server_params']['aggregator'],
        'content': read_py_module(
            f"./templates/aggregator/{config['server_params']['aggregator']}")
    }

    # read dataset_params.prep
    config['dataset_params']['prep'] = {
        'file': config['dataset_params']['prep'],
        'content': read_py_module(
            f"./templates/dataset_prep/{config['dataset_params']['prep']}")
    }

    return config
