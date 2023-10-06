'''
Implementation of FedAvg Aggregator for Server Aggregation.
'''
import torch


def aggregator(model, client_params: list, client_weights: list, kwargs=None):
    """
    Performs weighted sum of the client parameters, and returns the new model. 
    """

    # if any keyword arguments are passed
    _ = kwargs

    # zero out the model parameters weights
    set_to_zero_model_weights(model)

    # get the model parameters
    model_params = model.state_dict()

    # # enumerate over the clients
    # for k, client_param in enumerate(client_params):

    #     # enumerate over the layers of the model
    #     for layer_key in model_params:

    #         # skip layer if it is not float tensor, i.e., float32 and float64
    #         if not (model_params[layer_key].dtype == torch.float32 or model_params[layer_key].dtype == torch.float64):
    #             print('Skipping layer', layer_key)
    #             continue

    #         # calculate the weighted contribution from the client k
    #         # client_weights[k]
    #         contribution = client_param[layer_key].data * 1.0

    #         # add the contribution
    #         model_params[layer_key].data.add_(contribution)

    #     break

    print(client_params[0]['resnet50.layer1.0.bn3.num_batches_tracked'])

    model.load_state_dict(client_params[0])

    return model


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)
