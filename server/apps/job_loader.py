'''
This is the Job Leader Module, 
which is responsible for handling all training job 
related functionality.
'''
from apps.training_job import TrainingJobManager
from helpers.yaml import read_yaml_file
from helpers.logging import logger

# the job management dictionary
JOBS = dict()


def create_job(job_name: str, job_filename: str):
    config = read_yaml_file(f'./templates/job_config/{job_filename}')

    logger.info(f'Reading Job Config: \n{config}')

    # TODO: load all the python files into the configuration

    JOBS[job_name] = TrainingJobManager(project_name=job_name,
                                        client_params=config['client_params'],
                                        server_params=config['server_params'],
                                        dataset_params=config['dataset_params'])

    # add the clients from ClientManager to the job, to satisfy n_clients scenario else throw error
