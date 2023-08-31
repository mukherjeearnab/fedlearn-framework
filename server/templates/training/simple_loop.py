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
    The Training Loop Function. It trains the local_model of num_epochs.
    '''

    _ = global_model
    _ = extra_params

    # move the local_model to the device, cpu or gpu
    local_model = local_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

    # Epoch loop
    for _ in range(num_epochs):
        running_loss = 0.0

        # Training loop
        local_model.train()
        for _, (inputs, labels) in enumerate(train_loader, 0):

            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # report training loss here
        # if i % 100 == 99:
        #     print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}")
        #     running_loss = 0.0
