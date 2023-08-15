'''
Simple Neural Network Training Loop Module
'''
import torch
import torch.nn as nn
import torch.optim as optim


def train_loop(num_epochs: int, learning_rate: float,
               train_loader: torch.utils.data.dataloader.DataLoader,
               val_loader: torch.utils.data.dataloader.DataLoader,
               model) -> None:
    '''
    The Training Loop Function. It trains the model of num_epochs.
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Epoch loop
    for _ in range(num_epochs):
        running_loss = 0.0

        # Training loop
        model.train()
        for _, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # report training loss here
        # if i % 100 == 99:
        #     print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}")
        #     running_loss = 0.0

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # report validation accuracy here
        # print(
        #     f"Epoch {epoch+1} - Validation Accuracy: {100 * val_correct / val_total:.2f}%")
