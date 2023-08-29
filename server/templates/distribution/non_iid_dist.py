'''
Creates a random distribution of dataset of k clients.
returns as a list of tensors
'''
import torch


def distribute_into_client_chunks(dataset: tuple, client_weights: list) -> list:
    '''
    Creates random chunks by splitting the original dataset into 
    len(client_weights) chunks, based on the client weights.
    '''

    (data, labels) = dataset

    num_clients = len(client_weights)
    classes = torch.arange(10)

    # split the classes in a fair way
    classes_split = list()
    i = 0
    for n in range(num_clients):
        inc = i + int(client_weights[n]*len(classes))

        classes_split.append(classes[i:i+inc])

        i += inc

    client_chunks = list()
    for i in range(num_clients):
        idx = torch.stack([y_ == labels for y_ in classes_split[i]]).sum(
            0).bool()  # get indices for the classes
        client_chunks.append((data[idx], labels[idx]))

    return client_chunks
