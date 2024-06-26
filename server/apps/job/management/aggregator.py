'''
The Aggregator Module
'''

import os
import traceback
from time import sleep, time
from dotenv import load_dotenv
from helpers.argsparse import args
from helpers.logging import logger
from helpers.dynamod import load_module
from helpers.converters import convert_base64_to_state_dict
from helpers.converters import get_base64_state_dict
from helpers.torch import get_device
from helpers.file import torch_read
from helpers.converters import tensor_to_data_loader
from helpers.perflog import add_params, add_record, save_logs
from apps.job.api import get_job, allow_start_training, terminate_training, set_central_model_params, set_abort


# import environment variables
load_dotenv()


SERVER_PORT = int(args.port if args.port else os.getenv('SERVER_PORT'))
DELAY = float(os.getenv('DELAY'))


def aggregator_process(job_name: str, model):
    '''
    The aggregator process, running in background, and checking if ProcessPhase turns 2.
    If ProcessPhase is 2, run the aggregator function, and update the central model params,
    and set ProcessPhase to 1, by executing allow_start_training()
    '''

    logger.info(f'Starting Aggregator Thread for job {job_name}')

    device = get_device()

    # extra_data dict to store temporary training information
    extra_data = {}

    # curr_model = deepcopy(model)
    curr_model = model.to(device)

    sleep(DELAY*3)

    # retrieve the job instance
    state = get_job(job_name)
    logger.info(f'Retrieved Job State for Job {job_name}.')

    # start training
    allow_start_training(job_name)

    if 'hierarchical' in state and state['hierarchical']:
        client_config = state['client_params']
        client_split_key = 'splits'
        batch_size = state['client_params']['individual_configs'][0]['train_params']['batch_size']
    else:
        client_config = state['client_params']
        client_split_key = 'clients'
        batch_size = state['client_params']['train_params']['batch_size']

    CHUNK_DIR_NAME = 'dist'
    for chunk in client_config['dataset']['distribution'][client_split_key]:
        CHUNK_DIR_NAME += f'-{chunk}'

    # load the test dataset path
    DATASET_PREP_MOD = state['dataset_params']['prep']['file']
    DATASET_DIST_MOD = client_config['dataset']['distribution']['distributor']['file']
    DATASET_CHUNK_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"

    # load the test dataset from disk
    test_dataset = torch_read('global_test.tuple', DATASET_CHUNK_PATH)
    test_loader = tensor_to_data_loader(
        test_dataset, batch_size)

    # record start time
    start_time = time()

    i_agg_pp, i_agg_cs = -1, -1
    # keep listening to process_phase
    while True:
        # sleep for DELAY seconds
        sleep(DELAY)

        # get the current job state
        state = get_job(job_name)

        if state['job_status']['abort']:
            terminate_training(job_name)
            logger.info(
                f'Job [{job_name}] Aborted. Exiting Aggregator Process.')
            break

        if (i_agg_pp != state['job_status']['process_phase']) or (i_agg_cs != state['job_status']['client_stage']):
            logger.info(
                f"Checking for Aggregation Process for job [{job_name}] PS [{state['job_status']['process_phase']}] CS [{state['job_status']['client_stage']}]")
            i_agg_pp, i_agg_cs = state['job_status']['process_phase'], state['job_status']['client_stage']

        # if the process phase turns 2
        if state['job_status']['process_phase'] == 2 and state['job_status']['client_stage'] == 4:
            # log that aggregation is starting
            logger.info(f'Starting Aggregation Process for job [{job_name}]')

            # load the model module
            try:
                aggregator_module = load_module(
                    'agg_module', state['server_params']['aggregator']['content'])
            except Exception:
                logger.info(
                    f'Error loading Aggregator File. Terminating...\n{traceback.format_exc()}')
                set_abort(job_name)
                break

            # prepare the client params based on index
            param_state = get_job(job_name, params=True)
            client_params_ = param_state['exec_params']['client_model_params']
            client_params = [dict() for _ in client_params_]
            for client_param in client_params_:
                # retrieve the client params
                param = client_param['client_params']

                for i, client in enumerate(state['job_status']['client_info']):
                    if client['client_id'] == client_param['client_id']:
                        client_params[i] = convert_base64_to_state_dict(
                            param, device)

                        break

                # # retrieve the client index
                # index = int(client_param['client_id'].split('-')[1]) - 1

                # client_params[index] = convert_list_to_tensor(param)

            # run the aggregator function and obtain new global model
            try:
                curr_model = aggregator_module.aggregator(curr_model, client_params,
                                                          state['client_params']['dataset']['distribution'][client_split_key], extra_data, device, state['server_params']['train_params']['extra_params'])
            except Exception:
                logger.info(
                    f'Error Executing Aggregator. Terminating...\n{traceback.format_exc()}')
                set_abort(job_name)
                break

            # move to device, i.e., cpu or gpu
            curr_model = curr_model.to(device)

            # obtain the list form of model parameters
            params = get_base64_state_dict(curr_model)

            # update the central model params
            set_central_model_params(job_name, params)

            logger.info(
                f"Completed Global Round {state['job_status']['global_round']} out of {state['server_params']['train_params']['rounds']}")

            # logic to test the model with the aggregated parameters
            try:
                testing_module = load_module(
                    'testing_module', state['server_params']['test_file']['content'])
                metrics = testing_module.test_runner(
                    curr_model, test_loader, device)
            except Exception:
                logger.info(
                    f'Error Executing Model Tester. Terminating...\n{traceback.format_exc()}')
                set_abort(job_name)
                break

            # sleep(DELAY)

            # caclulate total time for 1 round
            end_time = time()
            # find the time delta for round and convert the microseconds to milliseconds
            time_delta = (end_time - start_time)*1000
            logger.info(f'Total Round Time Delta: {time_delta} ms')

            # PerfLog Methods
            add_params(job_name, state['job_status']['global_round'], params)
            add_record('server', job_name, metrics,
                       state['job_status']['global_round'], time_delta)
            save_logs(job_name)

            # set process phase to 1 to resume local training
            # check if global_round >= server_params.train_params.rounds, then terminate,
            # else allow training
            if state['job_status']['global_round'] >= state['server_params']['train_params']['rounds']:
                logger.info(f'Completed Job [{job_name}]. Terminating...')
                terminate_training(job_name)
                break
            else:
                allow_start_training(job_name)

                # record start time
                start_time = time()

            # log that aggregation is complete
            logger.info(f'Aggregation Process Complete for job [{job_name}]')

    logger.info(f'Terminating Aggregation Process for Job [{job_name}]')
