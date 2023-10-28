'''
Simple Neural Network Training Loop Module
'''
import torch
import torch.nn as nn
import torch.optim as optim


def train_loop(num_epochs: int, learning_rate: float,
               train_loader: torch.utils.data.dataloader.DataLoader,
               local_model, global_model,
               extra_params: dict, device='cpu') -> None:
    '''
    The Training Loop Function. It trains the model of num_epochs.
    '''

    # move the local_model to the device, cpu or gpu
    global_model = global_model.to(device)
    local_model = local_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

    # Epoch loop
    for epoch in range(num_epochs):
        local_model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):

            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)

            loss = criterion(outputs, labels)

            loss += (extra_params['fed_prox']['mu']/2) * \
                difference_models_norm_2(local_model, global_model)

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            print(f'Processing Batch {i}/{len(train_loader)}.', end='\r')

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum([torch.sum((tensor_1[i]-tensor_2[i])**2)
                for i in range(len(tensor_1))])

    return norm
