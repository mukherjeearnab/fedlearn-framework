'''
Model Testing Module
'''
import torch
from sklearn import metrics
from helpers.logging import logger


def test_runner(model, test_loader, device):
    '''
    The test orchestrator module. 
    1. Tests the Model against the test dataset.
    2. Reports the test results.
    '''

    results = test_model(model, test_loader, device)

    logger.info(
        f"Aggregated Model Report:\nLoss: {results['loss']:.4f}\nAccuracy: {results['accuracy']:.4f}\nF-1 Weighted: {results['f1_weighted']:.4f}\nROC-AUC Score: {results['roc_auc_score']:.4f}")

    return results


def test_model(model, test_loader, device) -> dict:
    '''
    Method to test a model in it's test dataset and return it's 
    performance metrics
    '''

    print('Starting Model Testing...')

    # move the model to the device, cpu or gpu and set to evaluation
    model = model.to(device)
    model.eval()

    print('Loaded Model to Device...')

    criterion = torch.nn.BCELoss()

    print('Starting Testing Loop...')

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0

        total_preds = []
        total_labels = []
        total_scores = []

        for i, _ in enumerate(test_loader, 1):
            print(i)

        print('In No Gradient Context')
        for i, (inputs, labels) in enumerate(test_loader, 1):
            print('starting to convert datasets to device')
            inputs, labels = inputs.to(device), labels.to(device)
            print('converted dataset to device')
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
            print('done with forward pass')

            predicted = (outputs > 0.5).float()

            print('computed predictions')

            total_labels.extend(labels.tolist())
            total_preds.extend(predicted.squeeze(1).tolist())
            total_scores.extend(outputs.squeeze(1).tolist())

            print(f'Processing batch {i} out of {len(test_loader)}')

        average_loss = val_loss / len(test_loader)

    print('\nDone Testing Model...')

    results = get_metrics(total_labels, total_preds, total_scores)

    results['loss'] = average_loss

    print('Done Calculating Metrics...')

    return results


def get_metrics(actuals: list, preds: list, scores: list) -> dict:
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
    confusion_matrix = metrics.confusion_matrix(actuals, preds).tolist()
    report = metrics.classification_report(actuals, preds)

    roc_auc_score = metrics.roc_auc_score(actuals, scores)

    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'precision_macro': precision_macro,
        'recall_weighted': recall_weighted,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix,
        'classification_report': report,
        'roc_auc_score': roc_auc_score
    }

    return results
