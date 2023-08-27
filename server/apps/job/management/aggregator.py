'''
The Aggregator Module
'''

import os
from time import sleep
from copy import deepcopy
from dotenv import load_dotenv
from helpers.logging import logger
from helpers.dynamod import load_module
from helpers.converters import convert_list_to_tensor
from helpers.converters import get_state_dict


# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
DELAY = int(os.getenv('DELAY'))


def aggregator_process(job_name: str, job_registry: dict, model):
    '''
    The aggregator process, running in background, and checking if ProcessPhase turns 2.
    If ProcessPhase is 2, run the aggregator function, and update the central model params,
    and set ProcessPhase to 1, by executing allow_start_training()
    '''

    logger.info(f'Starting Aggregator Thread for job {job_name}')

    curr_model = deepcopy(model)

    # retrieve the job instance
    job = job_registry[job_name]
    logger.info(f'Retrieved Job Instance for Job {job_name}.')

    # start training
    job.allow_start_training()

    # keep listening to process_phase
    while True:
        # sleep for DELAY seconds
        sleep(DELAY)

        # get the current job state
        state = job.get_state()

        logger.info(
            f"Checking for Aggregation Process for job [{job_name}] PS [{state['job_status']['process_phase']}] CS [{state['job_status']['client_stage']}]")

        # if the process phase turns 2
        if state['job_status']['process_phase'] == 2 and state['job_status']['client_stage'] == 4:
            # log that aggregation is starting
            logger.info(f'Starting Aggregation Process for job [{job_name}]')

            # load the model module
            aggregator_module = load_module(
                'agg_module', state['server_params']['aggregator']['content'])

            # prepare the client params based on index
            client_params_ = state['exec_params']['client_model_params']
            client_params = [dict() for _ in client_params_]
            for client_param in client_params_:
                # retrieve the client params
                param = client_param['client_params']

                for i, client in enumerate(state['exec_params']['client_info']):
                    if client['client_id'] == client_param['client_id']:
                        client_params[i] = convert_list_to_tensor(param)
                        break

                # # retrieve the client index
                # index = int(client_param['client_id'].split('-')[1]) - 1

                # client_params[index] = convert_list_to_tensor(param)

            # run the aggregator function and obtain new global model
            curr_model = aggregator_module.aggregator(curr_model, client_params,
                                                      state['client_params']['dataset']['distribution']['clients'])

            # obtain the list form of model parameters
            params = get_state_dict(curr_model)

            # update the central model params
            job.set_central_model_params(params)

            logger.info(
                f"Completed Global Round {job.job_status['global_round']-1} out of {job.server_params['train_params']['rounds']}")

            sleep(DELAY*3)

            # set process phase to 1 to resume local training
            # check if global_round >= server_params.train_params.rounds, then terminate,
            # else allow training
            if job.job_status['global_round'] > job.server_params['train_params']['rounds']:
                logger.info(f'Completed Job [{job_name}]. Terminating...')
                job.terminate_training()
                break
            else:
                job.allow_start_training()

            # log that aggregation is complete
            logger.info(f'Aggregation Process Complete for job [{job_name}]')

    logger.info(f'Terminating Aggregation Process for Job [{job_name}]')
