import torch
from scipy.integrate import simpson

from ..DeepLearning.Losses import mse, mae, bce, cce, Huber

"""
Calculates the values of different metrics based on training predictions and true values

data = (predictions, true_output)
metrics = [list of all wanted metrics]
loss = python function for the loss
"""
def calculate_metrics(data, metrics=(), loss=None, validation=False):
    val = "val_" if validation else ""
    values = {}
    predictions, Y = data
    for metric in metrics:
        found = True
        if metric == (val + "loss"):
            metric_value = loss(predictions, Y).item()
        elif metric == (val + "accuracy"):
            metric_value = accuracy(predictions, Y)
        elif metric == (val + "precision"):
            metric_value = precision(predictions, Y)
        elif metric == (val + "recall"):
            metric_value = recall(predictions, Y)
        elif metric == (val + "f1_score"):
            metric_value = f1_score(predictions, Y)
        elif metric == (val + "rmse"):
            metric_value = root_mean_squared_error(predictions, Y)
        elif metric == (val + "mae"):
            metric_value = mean_absolute_error(predictions, Y)
        elif metric == (val + "mse"):
            metric_value = mean_squared_error(predictions, Y)
        elif metric == (val + "bce"):
            metric_value = binary_cross_entropy(predictions, Y)
        elif metric == (val + "cce"):
            metric_value = categorical_cross_entropy(predictions, Y)
        elif metric == (val + "huber"):
            metric_value = huber_loss(predictions, Y)
        else:
            found = False
        if found: values[metric] = metric_value
    return values

def _round_dictionary(values):
    return {key: "{:0.4f}".format(value) for key, value in values.items()}


# ===============================CLASSIFICATION===============================
def accuracy(predictions, true_output):
    # if one-hot encoded
    if len(predictions.shape) == 2 and predictions.shape[1] > 1:
        predictions = torch.argmax(predictions, dim=1)
        true_output = torch.argmax(true_output, dim=1)
    else:
        # if true values {-1, 1} or {0, 1}
        if len(torch.unique(true_output)) == 2:
            if torch.any(true_output < 0):
                # true values {-1, 1}
                predictions = (predictions >= 0) * 2 - 1
            else:
                # true values {0, 1}
                predictions = predictions >= 0.5
    # now the values should be categorical and vectors of shape (n,)
    correct = predictions == true_output
    return correct.to(torch.float32).mean().item()

def precision(predictions, true_output):
    assert len(torch.unique(true_output)) == 2, "The problem must be binary classification."
    conf_mat = confusion_matrix(predictions, true_output)
    numerator = conf_mat[1, 1]
    denumenator = conf_mat[1, 1] + conf_mat[0, 1]
    return (numerator / denumenator).item()

def recall(predictions, true_output):
    assert len(torch.unique(true_output)) == 2, "The problem must be binary classification."
    conf_mat = confusion_matrix(predictions, true_output)
    numerator = conf_mat[1, 1]
    denumenator = conf_mat[1, 1] + conf_mat[1, 0]
    return (numerator / denumenator).item()

def roc_curve(probabilities, true_output, thresholds):
    assert len(torch.unique(true_output)) == 2, "The problem must be binary classification."
    tpr = torch.zeros_like(thresholds)
    fpr = torch.zeros_like(thresholds)
    for i, threshold in enumerate(reversed(thresholds)):
        predictions = _binary_prob_to_prediction(probabilities, threshold)
        conf_mat = confusion_matrix(predictions, true_output)
        tpr[i] = (conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])).item()
        fpr[i] = (conf_mat[0, 1] / (conf_mat[0, 1] + conf_mat[0, 0])).item()
    return fpr, tpr

def auc(fpr, tpr):
    if fpr[0] > fpr[-1]:
        fpr, tpr = reversed(fpr), reversed(tpr)
    return simpson(tpr, fpr)

def roc_auc(probabilities, true_output, thresholds):
    fpr, tpr = roc_curve(probabilities, true_output, thresholds)
    return auc(fpr, tpr)

def _binary_prob_to_prediction(probabilities, threshold=0.5):
    return (probabilities > threshold).squeeze().to(torch.int32)

def _one_hot_to_prediction(probabilities):
    return probabilities.argmax(dim=1)

def prob_to_pred(probabilities):
    if probabilities.ndim == 2:
        return _one_hot_to_prediction(probabilities)
    return _binary_prob_to_prediction(probabilities)



"""
Returns the confusion matrix of a problem with the smallest label in the top left corner.
"""
def confusion_matrix(predictions, true_output):
    classes = torch.unique(true_output).tolist()
    num_classes = len(classes)
    _confusion_matrix = torch.zeros((num_classes, num_classes))
    for pred, true in zip(predictions, true_output):
        j = classes.index(pred)
        i = classes.index(true)
        _confusion_matrix[i, j] += 1
    return _confusion_matrix

def f1_score(predictions, true_output):
    _precision = precision(predictions, true_output)
    _recall = recall(predictions, true_output)
    return (2 * _precision * _recall / (_precision + _recall))

def categorical_cross_entropy(predictions, true_output):
    return cce().loss(predictions, true_output).item()

def binary_cross_entropy(predictions, true_output):
    return bce().loss(predictions, true_output).item()


# ===============================REGRESSION===============================
def mean_squared_error(predictions, true_output):
    return mse().loss(predictions, true_output).item()

def root_mean_squared_error(predictions, true_output):
    return torch.sqrt(torch.mean((predictions - true_output) ** 2)).item()

def mean_absolute_error(predictions, true_output):
    return mae().loss(predictions, true_output).item()

def huber_loss(predictions, true_output):
    return Huber().loss(predictions, true_output).item()
