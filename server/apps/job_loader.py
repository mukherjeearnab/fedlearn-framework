'''
This is the Job Leader Module, 
which is responsible for handling all training job 
related functionality.
'''
import json
import os
from time import sleep
from copy import deepcopy
from multiprocessing import Process
from dotenv import load_dotenv
from apps.training_job import TrainingJobManager
from apps.client_api import get_alive_clients
from helpers.file import read_yaml_file, read_py_module, torch_write, torch_read
from helpers.file import set_OK_file, check_OK_file, create_dir_struct
from helpers.logging import logger
from helpers.http import get
from helpers.dynamod import load_module
from helpers.converters import convert_list_to_tensor
from helpers.converters import get_state_dict
# from apps.client_manager import ClientManager


# the client manager
# client_manager = ClientManager()

# # the job management dictionary
# JOBS = {}
# CONFIGS = {}


# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
DELAY = int(os.getenv('DELAY'))


def load_job(job_name: str, config_registry: dict):
    '''
    Load the Job Config and perform some preliminary validation.
    '''
    # load the job config file
    config = read_yaml_file(f'./templates/job_config/{job_name}.yaml')

    logger.info(f'Reading Job Config: \n{job_name}')

    config_registry[job_name] = config

    print('Job Config: ', json.dumps(config, sort_keys=True, indent=4))

    # get available clients from the server registry
    client_list = get_alive_clients()

    # check if sufficient clients are available or not, else return
    if len(client_list) < config['client_params']['num_clients']:
        logger.warning(f'''Not Enough Clients available on register. Please register more clients.
                     Required: {config['client_params']['num_clients']} Available: {len(client_list)}''')

    print('Job Config Loaded.')


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

    # load the python module files for the configuration
    config = load_module_files(config)
    logger.info('Loaded Py Modules from Job Config')

    # get available clients from the server registry
    client_list = get_alive_clients()

    # create a new job instance
    job_registry[job_name] = TrainingJobManager(project_name=job_name,
                                                client_params=config['client_params'],
                                                server_params=config['server_params'],
                                                dataset_params=config['dataset_params'])
    logger.info(f'Created Job Instance for Job {job_name}.')

    # assign the required number of clients to the job
    for i in range(config['client_params']['num_clients']):
        job_registry[job_name].add_client(client_list[i]['id'])
        logger.info(f'Assigning client to job {client_list[i]}')
    logger.info('Successfully assigned clients to job.')

    # # init calling the job manager route handler
    # get(f'http://localhost:{SERVER_PORT}/job_manager/init',
    #     {'job_id': job_name})

    # allow clients to download jobsheet
    job_registry[job_name].allow_jobsheet_download()
    logger.info(
        'Job sheet download set, waiting for clients to download and acknowledge.')

    ######################################################################################
    # wait for clients to ACK job sheet, i.e., wait until client stage becomes 1
    while True:
        logger.info(
            'Waiting for ClientStage to be [1] ClientReadyWithJobSheet')
        state = job_registry[job_name].get_state()

        if state['job_status']['client_stage'] == 1:
            break

        sleep(DELAY)
    ######################################################################################

    # prepare dataset for clients to download
    logger.info(f'Starting Dataset Preperation for Job {job_name}')
    prepare_dataset_for_deployment(config)
    logger.info(f'Dataset Preperation Complete for Job {job_name}')

    # set dataset download for clients
    job_registry[job_name].allow_dataset_download()

    logger.info('Allowing Clients to Download Dataset.')

    ##########################################
    # TODO: The logic below should run in a separate thread, since dataset pre times can be huge,
    # and may block the terminal main thread
    ##########################################

    ######################################################################################
    # wait for clients to ACK dataset, i.e., wait until client stage becomes 2
    while True:
        logger.info('Waiting for ClientStage to be [2] ClientReadyWithDataset')
        state = job_registry[job_name].get_state()

        if state['job_status']['client_stage'] == 2:
            break

        sleep(DELAY)
    ######################################################################################

    # get the initial model parameters
    params, model = load_model_and_get_params(config)

    # set the initial model parameters
    job_registry[job_name].set_central_model_params(params)

    # add process to listen to model process phase to change to 2 and start aggregation
    aggregator_proc = Process(target=aggregator_process,
                              args=(job_name, job_registry, model), name=f'aggregator_{job_name}')

    # start aggregation process
    aggregator_proc.start()

    # signal to start the training
    # JOBS[job_name].allow_start_training()

    # NOTE: CLIENTS WAIT FOR PROCESS PHASE TO CHANGE TO 2 AND THEN TO 1 AND THEN DOWNLOAD PARAMS
    return {
        'aggregator_proc': aggregator_proc, 'exec': True
    }


def load_module_files(config: dict) -> dict:
    '''
    This Module Loads the configuration files (.py modules) 
    as strings and adds them to the configuration dictionary.
    '''

    # read client_params.dataset.preprocessor
    config['client_params']['dataset']['preprocessor'] = {
        'file': config['client_params']['dataset']['preprocessor'],
        'content': read_py_module(
            f"./templates/preprocess/{config['client_params']['dataset']['preprocessor']}")
    }

    # read client_params.dataset.distribution.distributor
    config['client_params']['dataset']['distribution']['distributor'] = {
        'file': config['client_params']['dataset']['distribution']['distributor'],
        'content': read_py_module(
            f"./templates/distribution/{config['client_params']['dataset']['distribution']['distributor']}")
    }

    # read model_params.model_file
    config['client_params']['model_params']['model_file'] = {
        'file': config['client_params']['model_params']['model_file'],
        'content': read_py_module(
            f"./templates/models/{config['client_params']['model_params']['model_file']}")
    }

    # read model_params.parameter_mixer
    config['client_params']['model_params']['parameter_mixer'] = {
        'file': config['client_params']['model_params']['parameter_mixer'],
        'content': read_py_module(
            f"./templates/param_mixer/{config['client_params']['model_params']['parameter_mixer']}")
    }

    # read model_params.training_loop_file
    config['client_params']['model_params']['training_loop_file'] = {
        'file': config['client_params']['model_params']['training_loop_file'],
        'content': read_py_module(
            f"./templates/training/{config['client_params']['model_params']['training_loop_file']}")
    }

    # read model_params.test_file
    config['client_params']['model_params']['test_file'] = {
        'file': config['client_params']['model_params']['test_file'],
        'content': read_py_module(
            f"./templates/testing/{config['client_params']['model_params']['test_file']}")
    }

    # read server_params.aggregator
    config['server_params']['aggregator'] = {
        'file': config['server_params']['aggregator'],
        'content': read_py_module(
            f"./templates/aggregator/{config['server_params']['aggregator']}")
    }

    # read dataset_params.prep
    config['dataset_params']['prep'] = {
        'file': config['dataset_params']['prep'],
        'content': read_py_module(
            f"./templates/dataset_prep/{config['dataset_params']['prep']}")
    }

    return config


def prepare_dataset_for_deployment(config: dict):
    '''Prepares dataset for deployment
    1. run dataset_preperation and save to file in ./datasets/deploy/[dataset_prep file name]/root/dataset.tuple and OK file
    2. run dataset_distribution and save to file in ./datasets/deploy/[dataset_prep file name]/chunks/client-n-[client_weight].pt and OK file
    '''

    CHUNK_DIR_NAME = 'dist'
    for chunk in config['client_params']['dataset']['distribution']['clients']:
        CHUNK_DIR_NAME.join(f'-{chunk}')

    print('CHUNK_DIR_NAME', CHUNK_DIR_NAME)

    DATASET_ROOT_PATH = f"./datasets/deploy/{config['dataset_params']['prep']['file']}/root"
    DATASET_CHUNK_PATH = f"./datasets/deploy/{config['dataset_params']['prep']['file']}/chunks/{CHUNK_DIR_NAME}"

    # create the directory structures
    create_dir_struct(DATASET_ROOT_PATH)
    create_dir_struct(DATASET_CHUNK_PATH)

    # Step 1
    # if root dataset is not already present, prepare it
    if not check_OK_file(DATASET_ROOT_PATH):
        # load the dataset prep module
        dataset_prep_module = load_module(
            'dataset_prep', config['dataset_params']['prep']['content'])

        # obtain the dataset as data and labels
        data, labels = dataset_prep_module.prepare_dataset()

        # saving dataset file to disk
        try:
            # save the dataset to disk
            torch_write('dataset.tuple',
                        DATASET_ROOT_PATH,
                        (data, labels))

            # set the OK file
            set_OK_file(DATASET_ROOT_PATH)
            logger.info('Prepared Dataset Saved Successfully!')
        except Exception as e:
            logger.error(f'Error Saving Prepared Dataset to disk! {e}')
    else:
        logger.info('Root Dataset already present. Loading it from disk.')
        data, labels = torch_read('dataset.tuple', DATASET_ROOT_PATH)

    # Step 2
    # if chunk datasets are already not prepared, then prepare them
    if not check_OK_file(DATASET_CHUNK_PATH):
        # load the dataset prep module
        distributor_module = load_module(
            'distributor', config['client_params']['dataset']['distribution']['distributor']['content'])

        # obtain the dataset as data and labels
        chunks = distributor_module.distribute_into_client_chunks((data, labels),
                                                                  config['client_params']['dataset']['distribution']['clients'])

        # saving chunks to disk
        try:
            for i in range(config['client_params']['num_clients']):
                client = f'client-{i+1}'
                # save the dataset to disk
                torch_write(f'{client}.tuple',
                            DATASET_CHUNK_PATH,
                            chunks[i])

                logger.info(
                    f'Saved Chunk for {client} with size {len(chunks[i][1])}')

            # set the OK file
            set_OK_file(DATASET_CHUNK_PATH)
            logger.info('Dataset Client Chunks Saved Successfully!')
        except Exception as e:
            logger.error(f'Error Saving Chunked Dataset to disk! {e}')


def load_model_and_get_params(config: dict):
    '''
    Method to load the model and get initial parameters
    '''

    # load the model module
    model_module = load_module(
        'model_module', config['client_params']['model_params']['model_file']['content'])

    # create an instance of the model
    model = model_module.ModelClass()

    # obtain the list form of model parameters
    params = get_state_dict(model)

    return params, model


def aggregator_process(job_name: str, job_registry: dict, model):
    '''
    The aggregator process, running in background, and checking if ProcessPhase turns 2.
    If ProcessPhase is 2, run the aggregator function, and update the central model params,
    and set ProcessPhase to 1, by executing allow_start_training()
    '''

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

                # retrieve the client index
                index = int(client_param['client_id'].split('-')[1]) - 1

                client_params[index] = convert_list_to_tensor(param)

            # run the aggregator function and obtain new global model
            curr_model = aggregator_module.aggregator(curr_model, client_params,
                                                      state['client_params']['dataset']['distribution']['clients'])

            # obtain the list form of model parameters
            params = get_state_dict(curr_model)

            # update the central model params
            job.set_central_model_params(params)

            sleep(DELAY*3)

            # set process phase to 1 to resume local training
            job.allow_start_training()

            # log that aggregation is complete
            logger.info(f'Aggregation Process Complete for job [{job_name}]')
