'''
The Dataset Preperation Module for Jobs
'''
import traceback
import torch
from helpers.logging import logger
from helpers.file import torch_write, torch_read
from helpers.dynamod import load_module
from helpers.file import set_OK_file, check_OK_file, create_dir_struct, file_exists


def prepare_dataset_for_deployment(middleware_id: str, job_id: str, config: dict):
    '''Prepares dataset for deployment
    1. run dataset_preperation and save to file in ./datasets/deploy/[dataset_prep file name]/root/dataset.tuple and OK file
    2. run dataset_distribution and save to file in ./datasets/deploy/[dataset_prep file name]/chunks/[dist_file]/client-n-[client_weight].pt and OK file
    '''

    CHUNK_DIR_NAME = 'dist'
    for chunk in config['client_params']['dataset']['distribution']['clients']:
        CHUNK_DIR_NAME += f'-{chunk}'

    # print('CHUNK_DIR_NAME', CHUNK_DIR_NAME)
    DATASET_PREP_MOD = config['dataset_params']['prep']['file']
    DATASET_DIST_MOD = config['client_params']['dataset']['distribution']['distributor']['file']
    DATASET_ROOT_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/root"
    DATASET_CHUNK_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"

    # create the directory structures
    create_dir_struct(DATASET_ROOT_PATH)
    create_dir_struct(DATASET_CHUNK_PATH)

    # Step 1
    # if root dataset is not already present, prepare it
    # if not check_OK_file(DATASET_ROOT_PATH):

    file_name = f'{middleware_id}.tuple'
    dataset_path = f'./datasets/{job_id}'

    # obtain the dataset as data and labels
    (train_set, test_set) = torch_read(file_name, dataset_path)

    # saving dataset file to disk
    try:
        # save the train dataset to disk
        torch_write(f'{middleware_id}-train_dataset.tuple',
                    DATASET_ROOT_PATH, train_set)
        logger.info('Saved training dataset to disk!')

        # if test dataset is available, save it to disk
        if test_set is not None:
            torch_write(f'{middleware_id}-test_dataset.tuple',
                        DATASET_ROOT_PATH, test_set)
            logger.info('Saved testing dataset to disk!')

        # set the OK file
        set_OK_file(DATASET_ROOT_PATH)
        logger.info('Prepared Root Dataset Saved Successfully!')
    except Exception:
        logger.error(
            f'Error Saving Prepared Dataset to disk!\n{traceback.format_exc()}')
    # else:
    logger.info('Root Dataset already present. Loading it from disk.')
    train_set = torch_read(
        f'{middleware_id}-train_dataset.tuple', DATASET_ROOT_PATH)
    if file_exists(f'{DATASET_ROOT_PATH}/{middleware_id}-test_dataset.tuple'):
        test_set = torch_read(
            f'{middleware_id}-test_dataset.tuple', DATASET_ROOT_PATH)
    else:
        test_set = None

    # Step 2
    # if chunk datasets are already not prepared, then prepare them
    # if not check_OK_file(DATASET_CHUNK_PATH):
    # load the dataset prep module
    distributor_module = load_module(
        'distributor', config['client_params']['dataset']['distribution']['distributor']['content'])

    # obtain the dataset as data and labels
    train_chunks = distributor_module.distribute_into_client_chunks(train_set,
                                                                    config['client_params']['dataset']['distribution']['clients'])
    if test_set is not None:
        test_chunks = distributor_module.distribute_into_client_chunks(test_set,
                                                                       config['client_params']['dataset']['distribution']['clients'])

    # if test set is not available, split the chunks into train and test sets
    if test_set is None:
        split_ratio = list(
            config['client_params']['train_test_split'].values())
        chunks = [train_test_split(chunk, split_ratio)
                  for chunk in train_chunks]
    else:  # if test set is available, merge the train-test chunks into one
        chunks = list(zip(train_chunks, test_chunks))

    # create the global test set
    global_test_set = create_central_testset(
        [chunk[1] for chunk in chunks])

    # saving chunks and global test dataset to disk
    try:
        # save the chunks dataset to disk
        for i in range(config['client_params']['num_clients']):
            # client = f'client-{i+1}'
            # save the dataset to disk
            torch_write(f'{middleware_id}-{i}.tuple',
                        DATASET_CHUNK_PATH,
                        chunks[i])

            logger.info(
                f'Saved Chunk for {i+1}th Client with size {len(chunks[i][0][1])}, {len(chunks[i][1][1])}')

        # saving global test dataset to disk
        torch_write(f'{middleware_id}-global_test.tuple',
                    DATASET_CHUNK_PATH,
                    global_test_set)

        logger.info(
            f'Saved Global Test Set with size {len(global_test_set[1])}')

        # set the OK file
        set_OK_file(DATASET_CHUNK_PATH)
        logger.info(
            'Dataset Client Chunks and Global Set Saved Successfully!')
    except Exception:
        logger.error(
            f'Error Saving Chunked Dataset to disk!\n{traceback.format_exc()}')


def train_test_split(dataset: tuple, split_weights: list) -> tuple:
    '''
    Creates train and test sets by splitting the original dataset into 
    len(split_weights) chunks.
    '''

    data, labels = dataset

    total_data_samples = len(data)

    # calculate the split sections
    split_sections = [int(total_data_samples*weight)
                      for weight in split_weights]

    if sum(split_sections) < total_data_samples:
        split_sections[-1] += total_data_samples - sum(split_sections)
    else:
        split_sections[0] -= sum(split_sections) - total_data_samples

    # split the data and labels into chunks
    data_chunks = torch.split(data, split_size_or_sections=split_sections)
    label_chunks = torch.split(labels, split_size_or_sections=split_sections)

    # create dataset tuples for client chunks
    train_test_chunks = []
    for i in range(len(split_weights)):
        split_chunk = (data_chunks[i], label_chunks[i])

        train_test_chunks.append(split_chunk)

    # returns (train_set, test_set)
    return (train_test_chunks[0], train_test_chunks[1])


def create_central_testset(client_test_sets: list) -> tuple:
    '''
    Appendd all the client test sets into a single test set, 
    which will be used by the server to test the global model.
    '''
    # create a list of all the data and labels for each client
    data = [t[0] for t in client_test_sets]
    labels = [t[1] for t in client_test_sets]

    # concatenate the data and labels into a single tensor
    data = torch.cat(data, 0)
    labels = torch.cat(labels, 0)

    return (data, labels)
