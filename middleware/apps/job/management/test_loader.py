from apps.job.api import get_job
from helpers.file import torch_read
from helpers.converters import tensor_to_data_loader
from global_kvset import app_globals


def load_test_dataset(job_name: str, config: dict):
    '''
    Loads the Middleware's Global Test Set and return a DataLoader for the same
    '''
    config = get_job(job_name)

    app_globals.load()

    # load the test dataset
    DATASET_PREP_MOD = config['dataset_params']['prep']['file']
    DATASET_DIST_MOD = config['client_params']['dataset']['distribution']['distributor']['file']
    CHUNK_DIR_NAME = 'dist'
    for chunk in config['client_params']['dataset']['distribution']['clients']:
        CHUNK_DIR_NAME += f'-{chunk}'
    DATASET_CHUNK_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"
    # load the test dataset from disk
    test_dataset = torch_read(
        f'{app_globals.get("MIDDLEWARE_ID")}-global_test.tuple', DATASET_CHUNK_PATH)
    test_loader = tensor_to_data_loader(
        test_dataset, config['client_params']['train_params']['batch_size'])

    return test_loader
