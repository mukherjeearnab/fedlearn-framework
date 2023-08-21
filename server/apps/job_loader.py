'''
This is the Job Leader Module, 
which is responsible for handling all training job 
related functionality.
'''
from apps.training_job import TrainingJobManager
from helpers.file import read_yaml_file, read_py_module
from helpers.logging import logger
from apps.client_manager import ClientManager

# the client manager
client_manager = ClientManager()

# the job management dictionary
JOBS = dict()


def create_job(job_name: str, job_filename: str) -> None:
    '''
    Load a Job specification and create the job, and start initialization.
    '''
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

    # TODO: send request to data warehouse to start preparing the datasets and shards


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
