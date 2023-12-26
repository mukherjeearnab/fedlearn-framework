'''
Module Containing Download and Upload Methods for Server Comms
'''
import os
import traceback
from helpers.logging import logger
from helpers.http import get, post
from dotenv import load_dotenv

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def download_global_params(job_id: str, server_url: str):
    '''
    Method to download the global parameters for the training job
    '''

    global_params = None

    try:
        # listen to check if dataset flag is true or false
        url = f'{server_url}/job_manager/get_params'

        logger.info(
            f'Downloading Global Params for [{job_id}] from Server at {url}')

        manifest = get(url, {'job_id': job_id})

        global_params_dict = manifest['exec_params']['central_model_param']

        global_params = global_params_dict
    except Exception:
        logger.error(
            f'Failed to fetch job status state.\n{traceback.format_exc()}')

    return global_params


def upload_client_params(params: dict, client_id: str, job_id: str, server_url: str):
    '''
    Method to upload the trained client parameters to the server
    '''

    try:
        # listen to check if dataset flag is true or false
        url = f'{server_url}/job_manager/append_client_params'

        logger.info(
            f'Uploading Client Params for [{job_id}] from Server at {url}')

        post(url, {'client_id': client_id,
                   'client_params': params,
                   'job_id': job_id})

    except Exception:
        logger.error(
            f'Failed to fetch job status state.\n{traceback.format_exc()}')


def get_dataset_metadata(job_id: str, server_url: str):
    '''
    Method to download the global parameters for the training job
    '''

    try:
        # listen to check if dataset flag is true or false
        url = f'{server_url}/job_manager/get_dataset_metadata'

        logger.info(
            f'Downloading Dataset Metadata for [{job_id}] from Server at {url}')

        manifest = get(url, {'job_id': job_id})

        timestamp = manifest['timestamp']
        path = manifest['path']

        return timestamp, path
    except Exception:
        logger.error(
            f'Failed to fetch dataset metadata.\n{traceback.format_exc()}')

    return None, None
