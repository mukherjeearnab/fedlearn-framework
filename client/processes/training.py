'''
The Training Processes Module
'''
import torch
from helpers.file import torch_read, torch_write
from helpers.dynamod import load_module


def data_preprocessing(dataset_path: str, preprocessing_module_str: str, split_weights: list) -> tuple:
    '''
    Dataset Preprocessing Method.
    Takes input as the raw dataset path, and the preprocessing module as string.
    Loads the preprocessing module and processes the dataset after loading it from disk.
    '''

    # load the dataset and preprocessing module
    dataset = torch_read('dataset.tuple', dataset_path)
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
