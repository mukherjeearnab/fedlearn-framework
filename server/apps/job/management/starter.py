'''
The Job Starter Module
'''

import os
from time import sleep
from multiprocessing import Process
from dotenv import load_dotenv
from apps.job.exec_handler import JobExecHandler
from apps.job.param_handler import JobParamHandler
from apps.client.api import get_alive_clients
from apps.job.management.aggregator import aggregator_process
from helpers.logging import logger
from helpers.dynamod import load_module
from helpers.converters import get_base64_state_dict
from helpers.perflog import init_project
from helpers.torch import reset_seed

# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
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

    if 'hierarchical' in config and config['hierarchical']:
        client_params = config['middleware_params']
        hierarchical = True
        num_clients = len(config['middleware_params']['individual_configs'])
    else:
        client_params = config['client_params']
        hierarchical = False
        num_clients = config['client_params']['num_clients']

    # create a new job instance
    exec_handler = JobExecHandler(project_name=job_name,
                                  hierarchical=hierarchical,
                                  client_params=client_params,
                                  server_params=config['server_params'],
                                  dataset_params=config['dataset_params'])

    param_handler = JobParamHandler(project_name=job_name,
                                    hierarchical=hierarchical)

    job_registry[job_name] = (exec_handler, param_handler)

    logger.info(f'Created Job Instance for Job {job_name}.')

    # assign the required number of clients to the job
    for i in range(num_clients):
        exec_handler.add_client(
            client_list[i]['id'], client_list[i]['is_middleware'])
        logger.info(f'Assigning client to job {client_list[i]}')
    logger.info('Successfully assigned clients to job.')

    # # init calling the job manager route handler
    # get(f'http://localhost:{SERVER_PORT}/job_manager/init',
    #     {'job_id': job_name})

    # allow clients to download jobsheet
    exec_handler.allow_jobsheet_download()
    logger.info(
        'Job sheet download set, waiting for clients to download and acknowledge.')

    ######################################################################################
    # wait for clients to ACK job sheet, i.e., wait until client stage becomes 1
    while True:
        logger.info(
            'Waiting for ClientStage to be [1] ClientReadyWithJobSheet')
        state = exec_handler.get_state()

        if state['job_status']['client_stage'] == 1:
            break

        sleep(DELAY)
    ######################################################################################

    # set dataset download for clients
    exec_handler.allow_dataset_download()

    logger.info('Allowing Clients to Download Dataset.')

    ######################################################################################
    # wait for clients to ACK dataset, i.e., wait until client stage becomes 2
    while True:
        logger.info('Waiting for ClientStage to be [2] ClientReadyWithDataset')
        state = exec_handler.get_state()

        if state['job_status']['client_stage'] == 2:
            break

        sleep(DELAY)
    ######################################################################################

    # get the initial model parameters
    params, model = load_model_and_get_params(config)

    # set the initial model parameters
    param_handler.set_central_model_params(params)

    # init perflog project
    state = exec_handler.get_state()
    state['exec_params'] = param_handler.get_state()['exec_params']
    init_project(job_name, state)

    # add process to listen to model process phase to change to 2 and start aggregation
    aggregator_proc = Process(target=aggregator_process,
                              args=(job_name, model), name=f'aggregator_{job_name}')

    # start aggregation process
    aggregator_proc.start()

    # signal to start the training
    # JOBS[job_name].allow_start_training()

    # NOTE: CLIENTS WAIT FOR PROCESS PHASE TO CHANGE TO 2 AND THEN TO 1 AND THEN DOWNLOAD PARAMS
    return {
        'aggregator_proc': aggregator_proc, 'exec': True
    }


def load_model_and_get_params(config: dict):
    '''
    Method to load the model and get initial parameters
    '''

    # load the model module
    model_module = load_module(
        'model_module', config['server_params']['model_file']['content'])

    # reset torch seed
    reset_seed()

    # create an instance of the model
    model = model_module.ModelClass()

    # obtain the list form of model parameters
    params = get_base64_state_dict(model)

    return params, model
