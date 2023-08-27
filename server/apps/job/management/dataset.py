'''
The Dataset Preperation Module for Jobs
'''

from helpers.logging import logger
from helpers.file import torch_write, torch_read
from helpers.dynamod import load_module
from helpers.file import set_OK_file, check_OK_file, create_dir_struct


def prepare_dataset_for_deployment(config: dict):
    '''Prepares dataset for deployment
    1. run dataset_preperation and save to file in ./datasets/deploy/[dataset_prep file name]/root/dataset.tuple and OK file
    2. run dataset_distribution and save to file in ./datasets/deploy/[dataset_prep file name]/chunks/client-n-[client_weight].pt and OK file
    '''

    CHUNK_DIR_NAME = 'dist'
    for chunk in config['client_params']['dataset']['distribution']['clients']:
        CHUNK_DIR_NAME += f'-{chunk}'

    # print('CHUNK_DIR_NAME', CHUNK_DIR_NAME)

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
            logger.error(
                f'Error Saving Prepared Dataset to disk! {e}')
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
                # client = f'client-{i+1}'
                # save the dataset to disk
                torch_write(f'{i}.tuple',
                            DATASET_CHUNK_PATH,
                            chunks[i])

                logger.info(
                    f'Saved Chunk for {i+1}th Client with size {len(chunks[i][1])}')

            # set the OK file
            set_OK_file(DATASET_CHUNK_PATH)
            logger.info('Dataset Client Chunks Saved Successfully!')
        except Exception as e:
            logger.error(
                f'Error Saving Chunked Dataset to disk! {e}')
