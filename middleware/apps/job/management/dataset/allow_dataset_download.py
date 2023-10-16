import os
from time import sleep
from dotenv import load_dotenv
from helpers.logging import logger
from apps.job.api import get_job, allow_dataset_download

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))


def allow_downstream_dataset_download(job_name: str):
    '''
    Set the Download flag of the Middleware Chunk of Dataset for Downstream Clients
    '''

    allow_dataset_download(job_name)

    logger.info('Allowing Clients to Download Dataset.')

    ######################################################################################
    # wait for clients to ACK dataset, i.e., wait until client stage becomes 2
    i_cs = -1
    while True:
        state = get_job(job_name)

        if i_cs != state['job_status']['client_stage']:
            logger.info(
                'Waiting for ClientStage to be [2] ClientReadyWithDataset')
            i_cs = state['job_status']['client_stage']

        if state['job_status']['client_stage'] == 2:
            break

        sleep(DELAY)
    ######################################################################################
