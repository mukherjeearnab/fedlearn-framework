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
    w_dis = extra_params['mdis']['w_dis']
    update_cut = extra_params['mdis']['cut']
    update_threshold = extra_params['mdis']['cut_thresh']

    # extra_data
    extra_data['loss_history'] = {"prev": float(
        'inf'), "redn_count": 0} if 'loss_history' not in extra_data else extra_data['loss_history']
    extra_data['w_dis'] = w_dis if 'w_dis' not in extra_data else extra_data['w_dis']

    print(extra_data['w_dis'])

    # move the local_model to the device, cpu or gpu
    local_model = local_model.to(device)
    global_model = global_model.to(device)
    prev_local_model = prev_local_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    # Epoch loop
    for epoch in range(num_epochs):
        local_model.train()
        total_loss = 0.0

        w_mdis = 0.0 if epoch == 0 else extra_data['w_dis']

        for i, (inputs, labels) in enumerate(train_loader, 1):

            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            proj_l, pred_l = local_model.forward_with_projection(inputs)
            proj_g, _ = global_model.forward_with_projection(inputs)
            proj_pl, _ = prev_local_model.forward_with_projection(inputs)

            cposi = torch.pow(torch.norm((proj_l - proj_g)), 2)
            cnega = torch.pow(torch.norm((proj_l - proj_pl)), 2)

            con_loss = w_mdis * (cposi/cnega)

            sup_loss = criterion(pred_l, labels)

            loss = sup_loss + con_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f'Processing Batch {i}/{len(train_loader)}.', end='\r')

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

        if epoch == 0:
            if average_loss < extra_data['loss_history']['prev']:
                if extra_data['loss_history']['redn_count'] >= update_threshold:
                    extra_data['w_dis'] -= update_cut
                    if extra_data['w_dis'] < 0:
                        extra_data['w_dis'] = 0
                else:
                    extra_data['loss_history']['redn_count'] += 1
            else:
                extra_data['loss_history']['redn_count'] = 0
                extra_data['w_dis'] += update_cut

            extra_data['loss_history']['prev'] = average_loss
