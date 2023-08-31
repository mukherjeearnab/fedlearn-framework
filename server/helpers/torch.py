'''
Torch specific functionality module
'''
import os
import torch
from dotenv import load_dotenv

torch.manual_seed(0)

load_dotenv()
USE_CUDA = int(os.getenv('USE_CUDA'))


def get_device():
    '''
    Gets the device, gpu, if available else cpu
    '''

    if torch.cuda.is_available() and USE_CUDA == 1:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    return device
