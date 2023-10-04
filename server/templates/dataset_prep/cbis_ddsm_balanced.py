'''
Sample Dataset Preperation Module for MNSIT dataset
This module is executed before the client distribution 
for the dataset is performed in the Data Warehouse.
'''
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset():
    '''
    Prepare the MNIST Dataset here for Distribution to Clients
    NOTE: Returns the Train Set as the complete dataset.
    '''

    # Load and preprocess the dataset
    dataset = torch.load('/DATA/arnab/Projects/cbis-ddsm/total.tuple')
    data, labels = dataset

    return (data, labels)
