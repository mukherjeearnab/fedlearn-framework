'''
A Module with functions to handle and convert Tensors into floats and overall model management
'''
from collections import OrderedDict
import torch


def set_state_dict(model, state_dict: dict) -> None:
    '''
    Get the State Dict of a model as a Dictionart of List of Floats
    '''
    tensor_state_dict = convert_list_to_tensor(state_dict)
    model.load_state_dict(tensor_state_dict)


def get_state_dict(model) -> dict:
    '''
    Get the State Dict of a model as a Dictionart of List of Floats
    '''
    tensor_state_dict = model.state_dict()
    state_dict = convert_tensor_to_list(tensor_state_dict)
    return state_dict


def convert_list_to_tensor(params: OrderedDict) -> dict:
    '''
    Converts an OrderedDict of Tensors into a Dictionary of Lists of floats
    '''
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    '''
    Converts an Dictionary of List of floats into a Dictionary of Tensors
    '''
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_


def tensor_to_data_loader(dataset: tuple, batch_size: int):
    '''
    convert dataset tensor to data loader object
    '''

    data, labels = dataset

    train = data_utils.TensorDataset(data, labels)
    train_loader = data_utils.DataLoader(train, batch_size, shuffle=True)

    return train_loader
