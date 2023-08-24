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

    '''

    # Load and preprocess the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=True, transform=transform, download=True)
    # test_dataset = torchvision.datasets.MNIST(
    #     root='./datasets', train=False, transform=transform, download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    # test_data, test_labels = next(iter(test_loader))

    # join train set and test set together
    # data = torch.cat((train_data, test_data), 0)
    # labels = torch.cat((train_labels, test_labels), 0)

    data = train_data
    labels = train_labels

    return (data, labels)
