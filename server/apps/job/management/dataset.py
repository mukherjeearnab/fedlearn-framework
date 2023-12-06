'''
The Dataset Preperation Module for Jobs
'''
import torch
from helpers.logging import logger
from helpers.file import torch_write, torch_read
from helpers.dynamod import load_module
from helpers.file import set_OK_file, check_OK_file, create_dir_struct, file_exists


def prepare_dataset_for_deployment(config: dict):
    '''Prepares dataset for deployment
    1. run dataset_preperation and save to file in ./datasets/deploy/[dataset_prep file name]/root/dataset.tuple and OK file
    2. run dataset_distribution and save to file in ./datasets/deploy/[dataset_prep file name]/chunks/[dist_file]/client-n-[client_weight].pt and OK file
    '''

    if 'hierarchical' in config and config['hierarchical']:
        num_clients = len(config['middleware_params']['individual_configs'])
        client_config = config['middleware_params']
        client_split_key = 'splits'
    else:
        num_clients = config['client_params']['num_clients']
        client_config = config['client_params']
        client_split_key = 'clients'

    # Dynamic Distribution Flag, where distribution weights are not predefined
    dynamic_distribution = False
    if client_split_key not in client_config['dataset']['distribution']:
        dynamic_distribution = True
        client_config['dataset']['distribution'][client_split_key] = [
            0.1 for _ in range(num_clients)]

    CHUNK_DIR_NAME = 'dist'
    if dynamic_distribution:
        CHUNK_DIR_NAME += f'-{num_clients}'
    else:
        for chunk in client_config['dataset']['distribution'][client_split_key]:
            CHUNK_DIR_NAME += f'-{chunk}'

    # print('CHUNK_DIR_NAME', CHUNK_DIR_NAME)
    DATASET_PREP_MOD = config['dataset_params']['prep']['file']
    DATASET_DIST_MOD = client_config['dataset']['distribution']['distributor']['file']
    DATASET_ROOT_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/root"
    DATASET_CHUNK_PATH = f"./datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"

    # create the directory structures
    create_dir_struct(DATASET_ROOT_PATH)
    # dont create chunk dir id dynamic distribution
    if not dynamic_distribution:
        create_dir_struct(DATASET_CHUNK_PATH)

    # Step 1
    # if root dataset is not already present, prepare it
    if not check_OK_file(DATASET_ROOT_PATH):
        # load the dataset prep module
        dataset_prep_module = load_module(
            'dataset_prep', config['dataset_params']['prep']['content'])

        # obtain the dataset as data and labels
        (train_set, test_set) = dataset_prep_module.prepare_dataset()

        # saving dataset file to disk
        try:
            # save the train dataset to disk
            torch_write('train_dataset.tuple', DATASET_ROOT_PATH, train_set)
            logger.info('Saved training dataset to disk!')

            # if test dataset is available, save it to disk
            if test_set is not None:
                torch_write('test_dataset.tuple', DATASET_ROOT_PATH, test_set)
                logger.info('Saved testing dataset to disk!')

            # set the OK file
            set_OK_file(DATASET_ROOT_PATH)
            logger.info('Prepared Root Dataset Saved Successfully!')
        except Exception as e:
            logger.error(
                f'Error Saving Prepared Dataset to disk! {e}')
    else:
        logger.info('Root Dataset already present. Loading it from disk.')
        train_set = torch_read('train_dataset.tuple', DATASET_ROOT_PATH)
        if file_exists(f'{DATASET_ROOT_PATH}/test_dataset.tuple'):
            test_set = torch_read('test_dataset.tuple', DATASET_ROOT_PATH)
        else:
            test_set = None

    # Step 2
    # if chunk datasets are already not prepared, then prepare them
    if (not check_OK_file(DATASET_CHUNK_PATH)) or dynamic_distribution:
        # load the dataset prep module
        distributor_module = load_module(
            'distributor', client_config['dataset']['distribution']['distributor']['content'])

        # obtain the dataset as data and labels
        train_chunks, new_client_weights = distributor_module.distribute_into_client_chunks(train_set,
                                                                                            client_config['dataset']['distribution'][client_split_key], train=True)
        client_config['dataset']['distribution'][client_split_key] = new_client_weights

        # if dynamic distribution, update the dist chunk dir name with updated weights
        if dynamic_distribution:
            CHUNK_DIR_NAME_UPDATE = 'dist'
            for chunk in client_config['dataset']['distribution'][client_split_key]:
                CHUNK_DIR_NAME_UPDATE += f'-{chunk}'

            DATASET_CHUNK_PATH = DATASET_CHUNK_PATH.replace(
                CHUNK_DIR_NAME, CHUNK_DIR_NAME_UPDATE)
            create_dir_struct(DATASET_CHUNK_PATH)

        if test_set is not None:
            test_chunks, _ = distributor_module.distribute_into_client_chunks(test_set,
                                                                              client_config['dataset']['distribution'][client_split_key])

        # if test set is not available, split the chunks into train and test sets
        if test_set is None:
            split_ratio = list(
                client_config['train_test_split'].values())
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
            for i in range(num_clients):
                # client = f'client-{i+1}'
                # save the dataset to disk
                torch_write(f'{i}.tuple',
                            DATASET_CHUNK_PATH,
                            chunks[i])

                logger.info(
                    f'Saved Chunk for {i+1}th Client with size {len(chunks[i][0][1])}, {len(chunks[i][1][1])}')

            # saving global test dataset to disk
            torch_write('global_test.tuple',
                        DATASET_CHUNK_PATH,
                        global_test_set)

            logger.info(
                f'Saved Global Test Set with size {len(global_test_set[1])}')

            # set the OK file
            set_OK_file(DATASET_CHUNK_PATH)
            logger.info(
                'Dataset Client Chunks and Global Set Saved Successfully!')
        except Exception as e:
            logger.error(
                f'Error Saving Chunked Dataset to disk! {e}')


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
