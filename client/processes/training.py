'''
The Training Processes Module
'''
import torch
from helpers.file import torch_read
from helpers.dynamod import load_module
from helpers.logging import logger


def data_preprocessing(file_name:str, dataset_path: str, preprocessing_module_str: str, split_weights: list) -> tuple:
    '''
    Dataset Preprocessing Method.
    Takes input as the raw dataset path, and the preprocessing module as string.
    Loads the preprocessing module and processes the dataset after loading it from disk.
    '''

    # load the dataset and preprocessing module
    dataset = torch_read(file_name, dataset_path)
    preprocessor_module = load_module('preprocessor', preprocessing_module_str)

    # perform train and test split
    (train, test) = train_test_split(dataset, split_weights)

    # preprocess train and test datasets
    (train_processed, test_processed) = preprocessor_module.preprocess_dataset(train, test)

    return (train_processed, test_processed)


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

    # split the data and labels into chunks
    data_chunks = torch.split(data, split_size_or_sections=split_sections)
    label_chunks = torch.split(labels, split_size_or_sections=split_sections)

    # create dataset tuples for client chunks
    train_test_chunks = []
    for i in range(len(split_weights)):
        split_chunk = (data_chunks[i], label_chunks[i])

        train_test_chunks.append(split_chunk)

    # returns (train_set, test_set)
    return [train_test_chunks[0], train_test_chunks[1]]


def init_model(model_module_str: str):
    '''
    Initializes the model and returns it
    '''

    # load the model module
    model_module = load_module('model_module', model_module_str)

    # create an instance of the model
    model = model_module.ModelClass()

    return model


def parameter_mixing(current_global_params: dict, previous_local_params: dict, mixer_module: str) -> dict:
    '''
    Loads the parameter mixer and updates the parameters and returns it.
    '''

    # load the model module
    mixer_module = load_module('model_module', mixer_module)

    # create an instance of the model
    params = mixer_module.param_mixer(
        current_global_params, previous_local_params)

    return params


def train_model(job_manifest: dict, train_loader, model, device) -> dict:
    '''
    Execute the Training Loop
    '''

    logger.info('Starting Local Trainin with epochs')

    # load the train loop module
    train_loop_module = load_module('train_loop_module',
                                    job_manifest['client_params']['model_params']['training_loop_file']['content'])

    # assemble the hyperparameters
    num_epochs = job_manifest['client_params']['train_params']['local_epochs']
    learning_rate = job_manifest['client_params']['train_params']['learning_rate']

    logger.info(f'Set Epochs {num_epochs} and Learning Rate {learning_rate}')

    # train the model
    train_loop_module.train_loop(num_epochs, learning_rate, train_loader, [],
                                 model, device)
