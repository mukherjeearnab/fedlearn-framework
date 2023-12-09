'''
Simple Neural Network Training Loop Module
'''
import torch
import torch.nn as nn
import torch.optim as optim


def train_loop(num_epochs: int, learning_rate: float,
               train_loader: torch.utils.data.dataloader.DataLoader,
               local_model, global_model, prev_local_model,
               extra_params: dict, extra_data: dict, device='cpu') -> None:
    '''
    The Training Loop Function. It trains the local_model of num_epochs.
    '''

    # hyperparameters
    w_con = extra_params['conr']['w_con']
    w_sup = extra_params['conr']['w_sup']
    decay = extra_params['conr']['decay']

    # extra_data
    curr_round, total_round = float(extra_data['round_info']['current_round']), float(
        extra_data['round_info']['total_rounds'])
    decay_value = 1 - ((curr_round-1)/total_round) if decay else 1.0

    # move the local_model to the device, cpu or gpu
    local_model = local_model.to(device)
    global_model = global_model.to(device)
    _ = prev_local_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    cos = torch.nn.CosineSimilarity(dim=-1)

    # Epoch loop
    for epoch in range(num_epochs):
        local_model.train()
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 1):

            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            proj_l, pred_l = local_model.forward_with_projection(inputs)
            proj_g, _ = global_model.forward_with_projection(inputs)

            contrast = cos(proj_l, proj_g)

            con_loss = (decay_value * w_con) * torch.mean(1 - contrast)

            sup_loss = w_sup * criterion(pred_l, labels)

            loss = sup_loss + con_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f'Processing Batch {i}/{len(train_loader)}.', end='\r')

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
