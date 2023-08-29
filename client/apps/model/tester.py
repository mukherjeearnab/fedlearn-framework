'''
Model Testing Module
'''
import torch
from sklearn import metrics


def test_model(model, test_loader, device) -> dict:
    '''
    Method to test a model in it's test dataset and return it's 
    performance metrics
    '''

    # move the model to the device, cpu or gpu and set to evaluation
    model = model.to(device)
    model.eval()

    # the actual labels and predictions lists
    actuals = []
    preds = []

    with torch.no_grad():

        for inputs, labels in test_loader:
            # move tensors to the device, cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # val_total += labels.size(0)
            # val_correct += (predicted == labels).sum().item()

            actuals += labels.tolist()
            preds += predicted.tolist()

    results = get_metrics(actuals, preds)

    return results


def get_metrics(actuals: list, preds: list) -> dict:
    '''
    Returns a dictionary of evaluation metrics.
    accuracy, precision, recall, f-1 score, f-1 macro, f-1 micro, confusion matrix
    '''

    # print(actuals, preds)

    accuracy = metrics.accuracy_score(actuals, preds)
    precision_weighted = metrics.precision_score(actuals, preds,
                                                 average='weighted')
    precision_macro = metrics.precision_score(actuals, preds,
                                              average='macro')
    recall_weighted = metrics.recall_score(actuals, preds, average='weighted')
    recall_macro = metrics.recall_score(actuals, preds, average='macro')
    f1_macro = metrics.f1_score(actuals, preds, average='macro')
    f1_weighted = metrics.f1_score(actuals, preds, average='weighted')
    confusion_matrix = metrics.confusion_matrix(actuals, preds)
    report = metrics.classification_report(actuals, preds)

    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'precision_macro': precision_macro,
        'recall_weighted': recall_weighted,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix,
        'classification_report': report
    }

    return results
