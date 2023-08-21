'''
This is the Job Leader Module, 
which is responsible for handling all training job 
related functionality.
'''
import importlib
from apps.training_job import TrainingJobManager
from helpers.file import read_yaml_file, read_py_module, torch_write, set_OK_file, check_OK_file
from helpers.logging import logger
from helpers.dynamod import load_module
from apps.client_manager import ClientManager

# the client manager
client_manager = ClientManager()

# the job management dictionary
JOBS = dict()


def init_job(job_name: str, job_filename: str) -> None:
    '''
    Load a Job specification and create the job, and start initialization.
    '''
    # load the job config file
    config = read_yaml_file(f'./templates/job_config/{job_filename}')

    logger.info(f'Reading Job Config: \n{config}')

    # get available clients from the server registry
    client_list = client_manager.get_clients()

    # check if sufficient clients are available or not, else return
    if len(client_list) < config['client_params']['num_clients']:
        logger.error(f'''Not Enough Clients available on register. Please register more clients.
                     Required: {config['client_params']['num_clients']} Available: {len(client_list)}''')
        return

    # load the python module files for the configuration
    config = load_module_files(config)
    logger.info('Loaded Py Modules from Job Config')

    # create a new job instance
    JOBS[job_name] = TrainingJobManager(project_name=job_name,
                                        client_params=config['client_params'],
                                        server_params=config['server_params'],
                                        dataset_params=config['dataset_params'])
    logger.info(f'Created Job Instance for Job {job_name}.')

    # assign the required number of clients to the job
    for i in range(config['client_params']['num_clients']):
        JOBS[job_name].add_client(client_list[i]['name'])
        logger.info(f'Assigning client to job {client_list[i]}')
    logger.info('Successfully assigned clients to job.')

    # allow clients to download jobsheet
    JOBS[job_name].allow_jobsheet_download()
    logger.info(
        'Job sheet download set, waiting for clients to download and acknowledge.')

    # prepare dataset for clients to download
    logger.info(f'Starting Dataset Preperation for Job {job_name}')
    prepare_dataset_for_deployment(config)
    logger.info(f'Dataset Preperation Complete for Job {job_name}')
    
    # set dataset download for clients
    JOBS[job_name].allow_dataset_download()

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
    config['model_params']['model_file'] = {
        'file': config['model_params']['model_file'],
        'content': read_py_module(
            f"./templates/models/{config['model_params']['model_file']}")
    }

    # read model_params.parameter_mixer
    config['model_params']['parameter_mixer'] = {
        'file': config['model_params']['parameter_mixer'],
        'content': read_py_module(
            f"./templates/param_mixer/{config['model_params']['parameter_mixer']}")
    }

    # read model_params.training_loop_file
    config['model_params']['training_loop_file'] = {
        'file': config['model_params']['training_loop_file'],
        'content': read_py_module(
            f"./templates/training/{config['model_params']['training_loop_file']}")
    }

    # read model_params.test_file
    config['model_params']['test_file'] = {
        'file': config['model_params']['test_file'],
        'content': read_py_module(
            f"./templates/testing/{config['model_params']['test_file']}")
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
        CHUNK_DIR_NAME += f'-{chunk}'

    DATASET_ROOT_PATH = f"./datasets/deploy/{config['dataset_params']['prep']['content']}/root"
    DATASET_CHUNK_PATH = f"./datasets/deploy/{config['dataset_params']['prep']['content']}/chunks/{CHUNK_DIR_NAME}"

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
        except:
            logger.warning('Error Saving Prepared Dataset to disk!')

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
            for i in range(config['client_params']['n_clients']):
                client = f'client-{i+1}'
                # save the dataset to disk
                torch_write(f'{client}.tuple',
                            DATASET_CHUNK_PATH,
                            chunks[i])

            # set the OK file
            set_OK_file(DATASET_CHUNK_PATH)
            logger.info('Dataset Client Chunks Saved Successfully!')
        except:
            logger.warning('Error Saving Chunked Dataset to disk!')
    