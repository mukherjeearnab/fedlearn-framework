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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=True, transform=transform, download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

    # obtain the data and label tensors
    train_data, train_labels = next(iter(train_loader))

    data = train_data
    labels = train_labels

    return (data, labels)
