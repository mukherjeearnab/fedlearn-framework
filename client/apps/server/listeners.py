'''
Module Containing Server Listeners for Flags and Signals
'''
from time import sleep
import os
from helpers.logging import logger
from helpers.http import get
from dotenv import load_dotenv

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def listen_to_dataset_download_flag(job_id: str, server_url: str):
    '''
    Method to listen to server to wait for dataset to get prepared,
    and move on with downloading the dataset
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            manifest = get(url, {'job_id': job_id})

            logger.info(
                f"Got dataset download flag [{manifest['job_status']['download_dataset']}], Expecting[True] for [{job_id}] from Server at {url}")

            # if download dataset flag is true, break and exit
            if manifest['job_status']['download_dataset']:
                break

        except Exception as e:
            logger.error(
                f'Failed to fetch Dataset Download Flag. {e}')

        sleep(DELAY)


def listen_to_start_training(job_id: str, server_url: str):
    '''
    Method to listen to server to wait to start training,
    i.e., Process Phase to change to 1
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            manifest = get(url, {'job_id': job_id})

            logger.info(
                f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [1] for [{job_id}] from Server at {url}")

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 1:
                break

        except Exception as e:
            logger.error(
                f'Failed to fetch Process Phase. {e}')

        sleep(DELAY)


def listen_for_central_aggregation(job_id: str, server_url: str):
    '''
    Method to listen to server to wait for central aggregation,
    i.e., Process Phase to change to 2
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            manifest = get(url, {'job_id': job_id})

            logger.info(
                f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [2] for [{job_id}] from Server at {url}")

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 2:
                break

        except Exception as e:
            logger.error(f'Failed to Process Phase. {e}')

        sleep(DELAY)


def listen_to_client_stage(client_stage: int, job_id: str, server_url: str):
    '''
    Method to listen to client stage
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            manifest = get(url, {'job_id': job_id})

            logger.info(
                f"Got client Stage [{manifest['job_status']['client_stage']}], Expecting [{client_stage}] for [{job_id}] from Server at {url}")

            # if download dataset flag is true, break and exit
            if manifest['job_status']['client_stage'] == client_stage:
                break

        except Exception as e:
            logger.error(f'Failed to Client Stage. {e}')

        sleep(DELAY)


def listen_for_param_download_training(job_id: str, server_url: str) -> int:
    '''
    Method to listen to server to wait to start training or terminate,
    i.e., Process Phase to change to 1 or 3
    '''

    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get'

            manifest = get(url, {'job_id': job_id})

            logger.info(
                f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [1,3] for [{job_id}] from Server at {url}")

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 1 or manifest['job_status']['process_phase'] == 3:
                break

        except Exception as e:
            logger.error(
                f'Failed to fetch job status state. {e}')

        sleep(DELAY)

    return manifest['job_status']['process_phase']
