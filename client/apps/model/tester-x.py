'''
Model Testing Module
'''
import torch
from sklearn import metrics
from helpers.file import torch_read
from helpers.converters import tensor_to_data_loader
from helpers.logging import logger


def test_runner(test_file_name: str, test_dataset_path: str, batch_size: int, model, device):
    '''
    The test orchestrator module. 
    1. It loads the test dataset.
    2. Tests the Model against the test dataset.
    3. Reports the test results.
    '''
    test_dataset = torch_read(test_file_name, test_dataset_path)
    test_loader = tensor_to_data_loader(test_dataset, batch_size)

    results = test_model(model, test_loader, device)

    logger.info(
        f"Aggregated Model Report:\nLoss: {results['loss']:.4f}\nAccuracy: {results['accuracy']:.4f}\nF-1 Weighted: {results['f1_weighted']:.4f}\nROC-AUC Score: {results['roc_auc_score']:.4f}")

    return results


def test_model(model, test_loader, device) -> dict:
    '''
    Method to test a model in it's test dataset and return it's 
    performance metrics
    '''

    # move the model to the device, cpu or gpu and set to evaluation
    model = model.to(device)
    model.eval()

    criterion = torch.nn.BCELoss()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0

        total_preds = []
        total_labels = []
        total_scores = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()

            total_labels.extend(labels.tolist())
            total_preds.extend(predicted.squeeze(1).tolist())
            total_scores.extend(outputs.squeeze(1).tolist())

        average_loss = val_loss / len(test_loader)

    results = get_metrics(total_labels, total_preds, total_scores)

    results['loss'] = average_loss

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
