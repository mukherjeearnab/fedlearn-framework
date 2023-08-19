'''
Simple Testing Module that reports the Accuracy of the test results.
'''
import torch


def test_model(model, test_loader, device='cpu'):
    '''
    Tests the Model against the Test Dataset and reports the accuracy.
    '''

    # move the model to the device, cpu or gpu and set to evaluation
    model = model.to(device)
    model.eval()

    with torch.no_grad():

        for inputs, labels in test_loader:

            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        accuracy = val_correct / val_total
        print(f'Validation Accuracy: {accuracy}')

        return {
            'accuracy': accuracy
        }
