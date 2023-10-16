import os
from time import sleep
from dotenv import load_dotenv
from helpers.logging import logger
from apps.job.api import get_job

# import environment variables
load_dotenv()

DELAY = float(os.getenv('DELAY'))

CLIENT_STAGE = {
    0: 'Client Online',
    1: 'Client Ready With Jobsheet',
    2: 'Client Ready With Dataset',
    3: 'Client Busy In Training',
    4: 'Client Waiting For Params',
    5: 'Client Terminated'
}


def wait_for_client_stage(job_name: str, client_stage: int):
    '''
    Set the Download flag of the Middleware Chunk of Dataset for Downstream Clients
    '''

    ######################################################################################
    # wait for clients to ACK dataset, i.e., wait until client stage becomes 2
    i_cs = -1
    while True:
        state = get_job(job_name)

        if i_cs != state['job_status']['client_stage']:
            logger.info(
                f'Waiting for ClientStage to be [{client_stage}] {CLIENT_STAGE[client_stage]}')
            i_cs = state['job_status']['client_stage']

        if state['job_status']['client_stage'] == client_stage:
            break

        sleep(DELAY)
    ######################################################################################
