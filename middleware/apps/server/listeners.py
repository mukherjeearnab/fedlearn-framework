'''
Module Containing Server Listeners for Flags and Signals
'''
from time import sleep
import os
import traceback
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
    i_dd = -1
    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get_exec'

            manifest = get(url, {'job_id': job_id})

            if i_dd != manifest['job_status']['download_dataset']:
                logger.info(
                    f"Got dataset download flag [{manifest['job_status']['download_dataset']}], Expecting[True] for [{job_id}] from Server at {url}")
                i_dd = manifest['job_status']['download_dataset']

            # if download dataset flag is true, break and exit
            if manifest['job_status']['download_dataset']:
                break

        except Exception:
            logger.error(
                f'Failed to fetch Dataset Download Flag.\n{traceback.format_exc()}')

        sleep(DELAY)


def listen_to_start_training(job_id: str, server_url: str):
    '''
    Method to listen to server to wait to start training,
    i.e., Process Phase to change to 1
    '''

    i_pp = -1
    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get_exec'

            manifest = get(url, {'job_id': job_id})

            if i_pp != manifest['job_status']['process_phase']:
                logger.info(
                    f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [1] for [{job_id}] from Server at {url}")
                i_pp = manifest['job_status']['process_phase']

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 1:
                break

        except Exception:
            logger.error(
                f'Failed to fetch Process Phase.\n{traceback.format_exc()}')

        sleep(DELAY)


def listen_for_central_aggregation(job_id: str, server_url: str):
    '''
    Method to listen to server to wait for central aggregation,
    i.e., Process Phase to change to 2
    '''

    i_pp = -1
    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get_exec'

            manifest = get(url, {'job_id': job_id})

            if i_pp != manifest['job_status']['process_phase']:
                logger.info(
                    f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [2] for [{job_id}] from Server at {url}")
                i_pp = manifest['job_status']['process_phase']

            # if download dataset flag is true, break and exit
            if manifest['job_status']['process_phase'] == 2:
                break

        except Exception:
            logger.error(f'Failed to Process Phase.\n{traceback.format_exc()}')

        sleep(DELAY)


def listen_to_client_stage(client_stage: int, job_id: str, server_url: str):
    '''
    Method to listen to client stage
    '''
    i_cs = -1
    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get_exec'

            manifest = get(url, {'job_id': job_id})

            if i_cs != manifest['job_status']['client_stage']:
                logger.info(
                    f"Got client Stage [{manifest['job_status']['client_stage']}], Expecting [{client_stage}] for [{job_id}] from Server at {url}")
                i_cs = manifest['job_status']['client_stage']

            # if download dataset flag is true, break and exit
            if manifest['job_status']['client_stage'] == client_stage:
                break

        except Exception:
            logger.error(f'Failed to Client Stage.\n{traceback.format_exc()}')

        sleep(DELAY)


def listen_for_param_download_training(job_id: str, server_url: str, local_round: int) -> int:
    '''
    Method to listen to server to wait to start training or terminate,
    i.e., Process Phase to change to 1 or 3
    '''

    i_pp, i_gr = -1, -1
    while True:
        try:
            # listen to check if dataset flag is true or false
            url = f'{server_url}/job_manager/get_exec'

            manifest = get(url, {'job_id': job_id})

            if i_pp != manifest['job_status']['process_phase'] or i_gr != manifest['job_status']['global_round']:
                logger.info(
                    f"Got Process Phase [{manifest['job_status']['process_phase']}], Expecting [1,3] for [{job_id}] from Server at {url}")
                logger.info(
                    f"Got Global Round [{manifest['job_status']['global_round']}], Expecting [{local_round+1}] for [{job_id}] from Server at {url}")
                i_pp, i_gr = manifest['job_status']['process_phase'], manifest['job_status']['global_round']

            # if download dataset flag is true, break and exit
            if (manifest['job_status']['process_phase'] == 1 or manifest['job_status']['process_phase'] == 3) and manifest['job_status']['global_round'] == local_round+1:
                break

        except Exception:
            logger.error(
                f'Failed to fetch job status state.\n{traceback.format_exc()}')

        sleep(DELAY)

    return manifest['job_status']['process_phase'], manifest['job_status']['global_round'], manifest['job_status']['abort']
