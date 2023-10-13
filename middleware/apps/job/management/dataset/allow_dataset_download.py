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
    while True:
        logger.info('Waiting for ClientStage to be [2] ClientReadyWithDataset')
        state = get_job(job_name)

        if state['job_status']['client_stage'] == 2:
            break

        sleep(DELAY)
    ######################################################################################
