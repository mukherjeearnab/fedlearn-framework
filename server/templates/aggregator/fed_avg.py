'''
Implementation of FedAvg Aggregator for Server Aggregation.
'''
from copy import deepcopy


def aggregator(model, client_params: list, client_weights: list, kwargs=None):
    """
    Performs weighted sum of the client parameters, and returns the new model. 
    """

    # if any keyword arguments are passed
    _ = kwargs

    # create a new copy of the model to work on
    new_model = deepcopy(model)

    # zero out the model parameters weights
    set_to_zero_model_weights(new_model)

    model_params = new_model.state_dict()

    # enumerate over the clients
    for k, client_param in enumerate(client_params):

        # enumerate over the layers of the model
        for layer_key in model_params:

            # calculate the weighted contribution from the client k
            contribution = client_param[layer_key].data * client_weights[k]

            # add the contribution
            model_params[layer_key].data.add_(contribution)

    return new_model


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)
