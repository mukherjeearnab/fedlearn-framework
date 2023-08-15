'''
Sample Dataset Preprocessing Module using MNSIT dataset
'''
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size: int, train_ratio: float, test_ratio: float, validation_ratio: float):
    '''
    Prepare the Dataset here for training and evaluation
        1. Load the dataset.
        2. Split the dataset into training, validation and testing sets, based on the ratios in params.
        3. Balance the training set. (and the validation and testing sets, if required)
        4. Create DataLoaders for the three split datasets.
        5. Return the training, testing and validation datasets
    '''

    # Load and preprocess the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)

    # Split the training dataset into training, validation
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader,  val_loader
