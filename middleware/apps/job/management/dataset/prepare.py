from helpers.logging import logger
from apps.job.management.dataset import prepare_dataset_for_deployment


def dataset_prepare_for_downstream_clients(middleware_id: str, job_name: str, config: dict):
    '''
    Prepares Dataset Chunks for Downstream Clients
    '''
    # prepare dataset for clients to download
    logger.info(f'Starting Dataset Preperation for Job {job_name}')
    prepare_dataset_for_deployment(middleware_id, job_name, config)
    logger.info(f'Dataset Preperation Complete for Job {job_name}')
