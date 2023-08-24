'''
Torch specific functionality module
'''
import torch


def get_device():
    '''
    Gets the device, gpu, if available else cpu
    '''

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    return device
