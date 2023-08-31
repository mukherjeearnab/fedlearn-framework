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
from helpers.torch import get_device
from apps.model.testing import test_runner


# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
DELAY = float(os.getenv('DELAY'))


def aggregator_process(job_name: str, job_registry: dict, model):
    '''
    The aggregator process, running in background, and checking if ProcessPhase turns 2.
    If ProcessPhase is 2, run the aggregator function, and update the central model params,
    and set ProcessPhase to 1, by executing allow_start_training()
    '''

    logger.info(f'Starting Aggregator Thread for job {job_name}')

    device = get_device()

    curr_model = deepcopy(model)
    curr_model = curr_model.to(device)

    # retrieve the job instance
    job = job_registry[job_name]
    logger.info(f'Retrieved Job Instance for Job {job_name}.')

    # start training
    job.allow_start_training()

    i_agg_pp, i_agg_cs = -1, -1
    # keep listening to process_phase
    while True:
        # sleep for DELAY seconds
        sleep(DELAY)

        # get the current job state
        state = job.get_state()

        if (i_agg_pp != state['job_status']['process_phase']) or (i_agg_cs != state['job_status']['client_stage']):
            logger.info(
                f"Checking for Aggregation Process for job [{job_name}] PS [{state['job_status']['process_phase']}] CS [{state['job_status']['client_stage']}]")
            i_agg_pp, i_agg_cs = state['job_status']['process_phase'], state['job_status']['client_stage']

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
                        client_params[i] = convert_list_to_tensor(
                            param, device)

                        break

                # # retrieve the client index
                # index = int(client_param['client_id'].split('-')[1]) - 1

                # client_params[index] = convert_list_to_tensor(param)

            # run the aggregator function and obtain new global model
            curr_model = aggregator_module.aggregator(curr_model, client_params,
                                                      state['client_params']['dataset']['distribution']['clients'])

            # move to device, i.e., cpu or gpu
            curr_model = curr_model.to(device)

            # obtain the list form of model parameters
            params = get_state_dict(curr_model)

            # update the central model params
            job.set_central_model_params(params)

            logger.info(
                f"Completed Global Round {job.job_status['global_round']-1} out of {job.server_params['train_params']['rounds']}")

            # logic to test the model with the aggregated parameters
            DATASET_PREP_MOD = state['dataset_params']['prep']['file']
            DATASET_DIST_MOD = state['client_params']['dataset']['distribution']['distributor']['file']
            CHUNK_DIR_NAME = 'dist'
            for chunk in state['client_params']['dataset']['distribution']['clients']:
                CHUNK_DIR_NAME += f'-{chunk}'
            DATASET_CHUNK_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"
            test_runner('global_test.tuple', DATASET_CHUNK_PATH,
                        state['client_params']['train_params']['batch_size'],
                        curr_model, device)

            sleep(DELAY*6)

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
