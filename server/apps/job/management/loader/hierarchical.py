'''
Module Loader for Single Level Federated Learning
'''

from copy import deepcopy
from helpers.file import read_py_module


def load_module_files(config: dict) -> dict:
    '''
    This Module Loads the configuration files (.py modules) 
    as strings and adds them to the configuration dictionary.
    '''

    config = deepcopy(config)

    # read middleware_params.dataset.distribution.distributor
    config['middleware_params']['dataset']['distribution']['distributor'] = {
        'file': config['middleware_params']['dataset']['distribution']['distributor'],
        'content': read_py_module(
            f"./templates/distribution/{config['middleware_params']['dataset']['distribution']['distributor']}")
    }

    # recursively load middleware_params modules
    recursive_loader(config['middleware_params'])

    # read server_params.aggregator
    config['server_params']['aggregator'] = {
        'file': config['server_params']['aggregator'],
        'content': read_py_module(
            f"./templates/aggregator/{config['server_params']['aggregator']}")
    }

    # read server_params.model_file
    config['server_params']['model_file'] = {
        'file': config['server_params']['model_file'],
        'content': read_py_module(
            f"./templates/models/{config['server_params']['model_file']}")
    }

    # read server_params.test_file
    config['server_params']['test_file'] = {
        'file': config['server_params']['test_file'],
        'content': read_py_module(
            f"./templates/testing/{config['server_params']['test_file']}")
    }

    # read dataset_params.prep
    config['dataset_params']['prep'] = {
        'file': config['dataset_params']['prep'],
        'content': read_py_module(
            f"./templates/dataset_prep/{config['dataset_params']['prep']}")
    }

    return config


def recursive_loader(middleware_params: dict):
    '''
    Recursively Load the hierarchical structure configs for the middlewares
    '''
    for i, middleware in enumerate(middleware_params['individual_configs']):
        if 'individual_configs' in middleware:
            recursive_loader(middleware)

        middleware = deepcopy(middleware)

        # read client_params.aggregation.aggregator
        middleware['aggregation']['aggregator'] = {
            'file': middleware['aggregation']['aggregator'],
            'content': read_py_module(
                f"./templates/aggregator/{middleware['aggregation']['aggregator']}")
        }

        # read client_params.dataset.preprocessor
        middleware['dataset']['preprocessor'] = {
            'file': middleware['dataset']['preprocessor'],
            'content': read_py_module(
                f"./templates/preprocess/{middleware['dataset']['preprocessor']}")
        }

        # read client_params.dataset.distribution.distributor
        middleware['dataset']['distribution']['distributor'] = {
            'file': middleware['dataset']['distribution']['distributor'],
            'content': read_py_module(
                f"./templates/distribution/{middleware['dataset']['distribution']['distributor']}")
        }

        # read model_params.model_file
        middleware['model_params']['model_file'] = {
            'file': middleware['model_params']['model_file'],
            'content': read_py_module(
                f"./templates/models/{middleware['model_params']['model_file']}")
        }

        # read model_params.parameter_mixer
        middleware['model_params']['parameter_mixer'] = {
            'file': middleware['model_params']['parameter_mixer'],
            'content': read_py_module(
                f"./templates/param_mixer/{middleware['model_params']['parameter_mixer']}")
        }

        # read model_params.training_loop_file
        middleware['model_params']['training_loop_file'] = {
            'file': middleware['model_params']['training_loop_file'],
            'content': read_py_module(
                f"./templates/training/{middleware['model_params']['training_loop_file']}")
        }

        # read model_params.test_file
        middleware['model_params']['test_file'] = {
            'file': middleware['model_params']['test_file'],
            'content': read_py_module(
                f"./templates/testing/{middleware['model_params']['test_file']}")
        }

        middleware_params[i] = middleware
