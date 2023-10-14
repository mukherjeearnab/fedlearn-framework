'''
Module Loader for Single Level Federated Learning
'''

from helpers.file import read_py_module


def load_module_files(config: dict) -> dict:
    '''
    This Module Loads the configuration files (.py modules) 
    as strings and adds them to the configuration dictionary.
    '''

    # read client_params.dataset.preprocessor
    config['client_params']['dataset']['preprocessor'] = {
        'file': config['client_params']['dataset']['preprocessor'],
        'content': read_py_module(
            f"./templates/preprocess/{config['client_params']['dataset']['preprocessor']}")
    }

    # read client_params.dataset.distribution.distributor
    config['client_params']['dataset']['distribution']['distributor'] = {
        'file': config['client_params']['dataset']['distribution']['distributor'],
        'content': read_py_module(
            f"./templates/distribution/{config['client_params']['dataset']['distribution']['distributor']}")
    }

    # read model_params.model_file
    config['client_params']['model_params']['model_file'] = {
        'file': config['client_params']['model_params']['model_file'],
        'content': read_py_module(
            f"./templates/models/{config['client_params']['model_params']['model_file']}")
    }

    # read model_params.parameter_mixer
    config['client_params']['model_params']['parameter_mixer'] = {
        'file': config['client_params']['model_params']['parameter_mixer'],
        'content': read_py_module(
            f"./templates/param_mixer/{config['client_params']['model_params']['parameter_mixer']}")
    }

    # read model_params.training_loop_file
    config['client_params']['model_params']['training_loop_file'] = {
        'file': config['client_params']['model_params']['training_loop_file'],
        'content': read_py_module(
            f"./templates/training/{config['client_params']['model_params']['training_loop_file']}")
    }

    # read model_params.test_file
    config['client_params']['model_params']['test_file'] = {
        'file': config['client_params']['model_params']['test_file'],
        'content': read_py_module(
            f"./templates/testing/{config['client_params']['model_params']['test_file']}")
    }

    # read server_params.aggregator
    config['server_params']['aggregator'] = {
        'file': config['server_params']['aggregator'],
        'content': read_py_module(
            f"./templates/aggregator/{config['server_params']['aggregator']}")
    }

    # read dataset_params.prep
    config['dataset_params']['prep'] = {
        'file': config['dataset_params']['prep'],
        'content': read_py_module(
            f"./templates/dataset_prep/{config['dataset_params']['prep']}")
    }

    return config
