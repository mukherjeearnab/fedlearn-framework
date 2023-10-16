'''
The Job Starter Module
'''

import os
from time import sleep
from dotenv import load_dotenv
from apps.job.handlers.exec import JobExecHandler
from apps.job.handlers.param import JobParamHandler
from apps.client.api import get_alive_clients
from helpers.logging import logger

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def start_job(job_name: str, config_registry: dict, job_registry: dict) -> dict:
    '''
    Load a Job specification and create the job, and start initialization.
    return the dict containing the dict of additional information, which includes, the thread running the aggregator
    '''
    print('Starting Job.')

    if job_name not in config_registry:
        logger.error(f'Job Name: {job_name} is not loaded.')
        return {'exec': False}

    config = config_registry[job_name]

    # get available clients from the server registry
    client_list = get_alive_clients()

    # create a new job instance
    exec_handler = JobExecHandler(project_name=job_name,
                                  client_params=config['client_params'],
                                  server_params=config['server_params'],
                                  dataset_params=config['dataset_params'])

    param_handler = JobParamHandler(project_name=job_name)

    job_registry[job_name] = (exec_handler, param_handler)

    logger.info(f'Created Job Instance for Job {job_name}.')

    # assign the required number of clients to the job
    for i in range(config['client_params']['num_clients']):
        exec_handler.add_client(
            client_list[i]['id'], client_list[i]['is_middleware'])
        logger.info(f'Assigning client to job {client_list[i]}')
    logger.info('Successfully assigned clients to job.')

    # allow clients to download jobsheet
    exec_handler.allow_jobsheet_download()
    logger.info(
        'Job sheet download set, waiting for clients to download and acknowledge.')

    ######################################################################################
    # wait for clients to ACK job sheet, i.e., wait until client stage becomes 1
    i_cs = -1
    while True:
        state = exec_handler.get_state()

        if i_cs != state['job_status']['client_stage']:
            logger.info(
                'Waiting for ClientStage to be [1] ClientReadyWithJobSheet')
            i_cs = state['job_status']['client_stage']

        if state['job_status']['client_stage'] == 1:
            break

        sleep(DELAY)
    ######################################################################################
