from helpers.dynamod import load_module
from helpers.converters import convert_base64_to_state_dict
from helpers.converters import get_base64_state_dict
from apps.job.api import get_job


def aggregate_downstream_params(job_name: str, test_loader, device):
    # prepare the client params based on index
    state = get_job(job_name)
    param_state = get_job(job_name, params=True)
    client_params_ = param_state['exec_params']['client_model_params']
    client_params = [dict() for _ in client_params_]
    for client_param in client_params_:
        # retrieve the client params
        param = client_param['client_params']

        for i, client in enumerate(state['job_status']['client_info']):
            if client['client_id'] == client_param['client_id']:
                client_params[i] = convert_base64_to_state_dict(
                    param, device)

                break

        # # retrieve the client index
        # index = int(client_param['client_id'].split('-')[1]) - 1

        # client_params[index] = convert_list_to_tensor(param)

    # get the initial model parameters
    _, curr_model = load_model_and_get_params(state)

    curr_model = curr_model.to(device)

    # load the model module
    aggregator_module = load_module(
        'agg_module', state['server_params']['aggregator']['content'])

    # run the aggregator function and obtain new global model
    curr_model = aggregator_module.aggregator(curr_model, client_params,
                                              state['client_params']['dataset']['distribution']['clients'])

    # move to device, i.e., cpu or gpu
    curr_model = curr_model.to(device)

    # logic to test the model with the aggregated parameters
    testing_module = load_module(
        'testing_module', state['client_params']['model_params']['test_file']['content'])
    metrics = testing_module.test_runner(
        curr_model, test_loader, device)

    # obtain the list form of model parameters
    params = get_base64_state_dict(curr_model)

    return params, metrics


def load_model_and_get_params(config: dict):
    '''
    Method to load the model and get initial parameters
    '''

    # load the model module
    model_module = load_module(
        'model_module', config['client_params']['model_params']['model_file']['content'])

    # # reset torch seed
    # reset_seed()

    # create an instance of the model
    model = model_module.ModelClass()

    # obtain the list form of model parameters
    params = get_base64_state_dict(model)

    return params, model
