'''
Module Containing Server Listeners for Flags and Signals
'''
from time import sleep
from helpers.logging import logger
from helpers.http import get


def listen_to_dataset_download_flag(job_id: str, server_url: str):
    '''
    Method to listen to server to wait for dataset to get prepared,
    and move on with downloading the dataset
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            logger.info(
                f'Fetching Job Manifest for [{job_id}] from Server at {url}')

            manifest = get(url, {'job_id': job_id})

            # if download dataset flag is true, break and exit
            if manifest['job_status']['download_dataset']:
                break

        except:
            logger.warning(f'Failed to fetch job status state.')

        sleep(5)


def listen_to_start_training(job_id: str, server_url: str):
    '''
    Method to listen to server to wait to start training,
    i.e., Process Phase to change to 1
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            logger.info(
                f'Fetching Job Manifest for [{job_id}] from Server at {url}')

            manifest = get(url, {'job_id': job_id})

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 1:
                break

        except:
            logger.warning(f'Failed to fetch job status state.')

        sleep(5)
