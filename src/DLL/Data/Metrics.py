import torch
from math import sqrt

from ..DeepLearning.Losses import MSE, MAE, BCE, CCE, Huber, Exponential


def calculate_metrics(data, metrics, loss=None, validation=False):
    """
    Calculates the values of different metrics based on training predictions and true values.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): A tuple of predictions and the true outputs of a model.
        metrics (tuple | list): A list metric names to calculate. Each element must be in ["loss", "accuracy", "precision", "recall", "f1_score", "rmse", "mae", "mse", "bce", "cce", "huber"].
        loss (Callable[[predictions, true values], float], optional): The wanted loss function. If Defaults to None.
        validation (bool, optional): Determines if "val_" is appended before each metric. If true, the each element of the metrics must be for instance "val_loss" or "val_mse". Defaults to False.

    Returns:
        dict[str, float]: A dictionary with metric name as the key and the metric as the value.
    """
    if not isinstance(data, tuple) or len(data) != 2:
        raise TypeError("data must be a tuple of length 2.")
    predictions, true_output = data
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("The elements of data must be torch tensors.")
    if true_output.shape != predictions.shape:
        raise ValueError("The shapes of the predictions and the output do not match.")
    if true_output.ndim > 2 or predictions.ndim > 2:
        raise ValueError(f"The shapes of the data are not 1 or 2 dimensional. Currently {true_output.ndim, predictions.ndim}.")
    available_metrics = ["loss", "accuracy", "precision", "recall", "f1_score", "rmse", "mae", "mse", "bce", "cce", "huber"]
    if any([(metric[4:] if validation else metric) not in available_metrics for metric in metrics]):
        raise ValueError(f"Only the following metrics are supported {available_metrics}. Currently {metrics}.")
    if ("loss" in metrics or "val_loss" in metrics) and loss is None:
        raise ValueError("For calculating the loss, the loss function must be passed as an argument.")

    val = "val_" if validation else ""
    values = {}
    for metric in metrics:
        if metric == (val + "loss") and loss is not None:
            metric_value = loss(predictions, true_output).item()
        elif metric == (val + "accuracy"):
            metric_value = accuracy(predictions, true_output)
        elif metric == (val + "precision"):
            metric_value = precision(predictions, true_output)
        elif metric == (val + "recall"):
            metric_value = recall(predictions, true_output)
        elif metric == (val + "f1_score"):
            metric_value = f1_score(predictions, true_output)
        elif metric == (val + "rmse"):
            metric_value = root_mean_squared_error(predictions, true_output)
        elif metric == (val + "mae"):
            metric_value = mean_absolute_error(predictions, true_output)
        elif metric == (val + "mse"):
            metric_value = mean_squared_error(predictions, true_output)
        elif metric == (val + "bce"):
            metric_value = binary_cross_entropy(predictions, true_output)
        elif metric == (val + "cce"):
            metric_value = categorical_cross_entropy(predictions, true_output)
        elif metric == (val + "huber"):
            metric_value = huber_loss(predictions, true_output)
        values[metric] = metric_value
    return values

def _round_dictionary(values):
    if not isinstance(values, dict):
        raise TypeError("values must be a dictionary")

    return {key: "{:0.4f}".format(value) for key, value in values.items()}


# ===============================CLASSIFICATION===============================
def accuracy(predictions, true_output):
    """
    Calculates the accuracy of predictions.

    Args:
        predictions (torch.Tensor): A torch tensor of either predicted labels or probabilities. If is given probabilities, calculates the corresponding predictions.
        true_output (torch.Tensor): A torch tensor of either true labels or one-hot encoded values. If is given a one-hot encoded tensor, calculates the corresponding labels.

    Returns:
        float: the accuracy of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim > 2:
        raise ValueError("predictions must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 2:
        raise ValueError("true_output must be a 1 or 2 dimensional torch tensor.")
    if len(predictions) != len(true_output):
        raise ValueError("predictions and true_output must have the same number of samples.")

    if predictions.ndim == 2 or torch.any(torch.bitwise_and(predictions > 0, predictions < 1)):
        predictions = prob_to_pred(predictions)
    if true_output.ndim == 2:
        true_output = prob_to_pred(true_output)
    correct = predictions == true_output
    return correct.to(torch.float32).mean().item()

def precision(predictions, true_output):
    """
    Calculates the precision or the positive predictive value of the predictions. The problem must be binary classification to be able to calculate the precision.

    Args:
        predictions (torch.Tensor of shape (n_samples,)): A torch tensor of either predicted labels or probabilities. If is given probabilities, calculates the corresponding predictions.
        true_output (torch.Tensor of shape (n_samples,)): A torch tensor of true labels.

    Returns:
        float: the precision of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim > 2:
        raise ValueError("predictions must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 1:
        raise ValueError("true_output must be a 1 dimensional torch tensor.")
    if len(predictions) != len(true_output):
        raise ValueError("predictions and true_output must have the same number of samples.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")

    if set(torch.unique(predictions).numpy()) != {0, 1}:
        predictions = prob_to_pred(predictions)
    conf_mat = confusion_matrix(predictions, true_output)
    numerator = conf_mat[1, 1]
    denumenator = conf_mat[1, 1] + conf_mat[0, 1]
    return (numerator / denumenator).item()

def recall(predictions, true_output):
    """
    Calculates the recall or the sensitivity of the predictions. The problem must be binary classification to be able to calculate the recall.

    Args:
        predictions (torch.Tensor of shape (n_samples,)): A torch tensor of either predicted labels or probabilities. If is given probabilities, calculates the corresponding predictions.
        true_output (torch.Tensor of shape (n_samples,)): A torch tensor of true labels.

    Returns:
        float: the recall of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim > 2:
        raise ValueError("predictions must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 1:
        raise ValueError("true_output must be a 1 dimensional torch tensor.")
    if len(predictions) != len(true_output):
        raise ValueError("predictions and true_output must have the same number of samples.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")

    if set(torch.unique(predictions).numpy()) != {0, 1}:
        predictions = prob_to_pred(predictions)
    conf_mat = confusion_matrix(predictions, true_output)
    numerator = conf_mat[1, 1]
    denumenator = conf_mat[1, 1] + conf_mat[1, 0]
    return (numerator / denumenator).item()

def roc_curve(probabilities, true_output, thresholds):
    """
    Calculates receiver operating characteristic curve. The problem must be binary classification.

    Args:
        probabilities (torch.Tensor of shape (n_samples,)): A torch tensor of probabilities.
        true_output (torch.Tensor of shape (n_samples,)): A torch tensor of true labels.
        thresholds (torch.Tensor of shape (n_thresholds,)): A tensor of thresholds. Every value must be between 0 and 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of false-positive-rate and true-positive-rate evaluated at the given thresholds.
    """
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim > 2:
        raise ValueError("probabilities must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 1:
        raise ValueError("true_output must be a 1 dimensional torch tensor.")
    if len(probabilities) != len(true_output):
        raise ValueError("probabilities and true_output must have the same number of samples.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")
    if not isinstance(thresholds, torch.Tensor):
        raise TypeError("thresholds must be a torch tensor.")
    if torch.any(torch.bitwise_or(thresholds < 0, thresholds > 1)):
        raise ValueError("thresholds must have every value must be between 0 and 1.")

    tpr = torch.zeros_like(thresholds)
    fpr = torch.zeros_like(thresholds)
    for i, threshold in enumerate(reversed(thresholds)):
        predictions = _binary_prob_to_prediction(probabilities, threshold)
        conf_mat = confusion_matrix(predictions, true_output)
        tpr[i] = (conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])).item()
        fpr[i] = (conf_mat[0, 1] / (conf_mat[0, 1] + conf_mat[0, 0])).item()
    return fpr, tpr

def auc(fpr, tpr):
    """
    Calculates area under the roc curve using the trapezoidal rule. The problem must be binary classification.

    Args:
        fpr (torch.Tensor of shape (n_thresholds,)): A torch tensor containing the false positive rates.
        tpr (torch.Tensor of shape (n_thresholds,)): A torch tensor containing the true positive rates.

    Returns:
        float: the area under the roc curve.
    """
    if not isinstance(fpr, torch.Tensor) or fpr.ndim > 1:
        raise ValueError("fpr must be a 1 dimensional torch tensor.")
    if not isinstance(tpr, torch.Tensor) or fpr.ndim > 1:
        raise ValueError("tpr must be a 1 dimensional torch tensor.")
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have the same number of thresholds.")

    indicies = torch.argsort(fpr)
    tpr = tpr[indicies]
    fpr = fpr[indicies]
    diffs = fpr[1:] - fpr[:-1]
    return torch.sum(diffs * (tpr[1:] + tpr[:-1]) / 2).item()

def roc_auc(probabilities, true_output, thresholds):
    """
    Calculates area under the roc curve. The problem must be binary classification.

    Args:
        probabilities (torch.Tensor of shape (n_samples,)): A torch tensor of probabilities.
        true_output (torch.Tensor of shape (n_samples,)): A torch tensor of true labels.
        thresholds (torch.Tensor of shape (n_thresholds,)): A tensor of thresholds. Every value must be between 0 and 1.

    Returns:
        float: the area under the roc curve.
    """
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim > 2:
        raise ValueError("probabilities must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 1:
        raise ValueError("true_output must be a 1 dimensional torch tensor.")
    if len(probabilities) != len(true_output):
        raise ValueError("probabilities and true_output must have the same number of samples.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")
    if not isinstance(thresholds, torch.Tensor):
        raise TypeError("thresholds must be a torch tensor.")
    if torch.any(torch.bitwise_or(thresholds < 0, thresholds > 1)):
        raise ValueError("thresholds must have every value must be between 0 and 1.")

    fpr, tpr = roc_curve(probabilities, true_output, thresholds)
    return auc(fpr, tpr)

def _binary_prob_to_prediction(probabilities, threshold=0.5):
    return (probabilities > threshold).to(torch.int32)

def _one_hot_to_prediction(probabilities):
    return probabilities.argmax(dim=1)

def prob_to_pred(probabilities):
    """
    Converts probabilities to predicted labels.

    Args:
        probabilities (torch.Tensor with 1 or 2 dimensions): A tensor of probabilities. Must either be 1 or 2 dimensional.
    """
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim > 2:
        raise ValueError("probabilities must be a 1 or 2 dimensional torch tensor.")

    if probabilities.ndim == 2:
        return _one_hot_to_prediction(probabilities)
    return _binary_prob_to_prediction(probabilities)

"""
Returns the confusion matrix of a problem with the smallest label in the top left corner.
"""
def confusion_matrix(predictions, true_output):
    """
    The confusion matrix. At the moment, the problem must be binary classification. The element at (i, j) represents the number of observations in class i predicted to be class j.

    Args:
        predictions (torch.Tensor): A torch tensor of either predicted labels or probabilities. If is given probabilities, calculates the corresponding predictions.
        true_output (torch.Tensor): A torch tensor of either true labels or one-hot encoded values. If is given a one-hot encoded tensor, calculates the corresponding labels.

    Returns:
        torch.Tensor of shape (n_classes, n_classes): The confusion matrix.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim > 2:
        raise ValueError("predictions must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 2:
        raise ValueError("true_output must be a 1 or 2 dimensional torch tensor.")
    if len(predictions) != len(true_output):
        raise ValueError("predictions and true_output must have the same number of samples.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")
    
    if predictions.ndim == 2 or torch.any(torch.bitwise_and(predictions > 0, predictions < 1)):
        predictions = prob_to_pred(predictions)
    if true_output.ndim == 2:
        true_output = prob_to_pred(true_output)

    classes = torch.unique(true_output).tolist()
    num_classes = len(classes)
    _confusion_matrix = torch.zeros((num_classes, num_classes))
    for pred, true in zip(predictions, true_output):
        j = classes.index(pred)
        i = classes.index(true)
        _confusion_matrix[i, j] += 1
    return _confusion_matrix

def f1_score(predictions, true_output):
    """
    Calculates the f1 score of the predictions. The problem must be binary classification.

    Args:
        predictions (torch.Tensor of shape (n_samples,)): A torch tensor of either predicted labels or probabilities. If is given probabilities, calculates the corresponding predictions.
        true_output (torch.Tensor of shape (n_samples,)): A torch tensor of true labels.

    Returns:
        float: the f1 score of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim > 2:
        raise ValueError("predictions must be a 1 or 2 dimensional torch tensor.")
    if not isinstance(true_output, torch.Tensor) or true_output.ndim > 1:
        raise ValueError("true_output must be a 1 dimensional torch tensor.")
    if len(predictions) != len(true_output):
        raise ValueError("predictions and true_output must have the same number of samples.")
    if set(torch.unique(predictions).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")

    if set(torch.unique(predictions).numpy()) != {0, 1}:
        predictions = prob_to_pred(predictions)
    _precision = precision(predictions, true_output)
    _recall = recall(predictions, true_output)
    return (2 * _precision * _recall / (_precision + _recall))

def categorical_cross_entropy(predictions, true_output):
    """
    Calculates the categorical cross entropy of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted values as a probability distribution. Must be the same shape as the true_output.
        true_output (torch.Tensor): A one-hot encoded tensor of true values. Must be the same shape as the prediction.

    Returns:
        float: The categorical cross entropy of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 2:
        raise ValueError("The predictions and the true output must be one-hot encoded.")
    
    return CCE().loss(predictions, true_output).item()

def binary_cross_entropy(predictions, true_output):
    """
    Calculates the binary cross entropy of the predictions. The problem must be binary classification.

    Args:
        predictions (torch.Tensor): A tensor of predicted probabilities as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true labels as a vector. Must be the same shape as the predictions.

    Returns:
        float: The binary cross entropy of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")
    
    return BCE().loss(predictions, true_output).item()

def exponential_loss(predictions, true_output):
    """
    Calculates the exponential loss of the predictions. The problem must be binary classification.

    Args:
        predictions (torch.Tensor): A tensor of predicted probabilities as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true labels as a vector. Must be the same shape as the predictions.

    Returns:
        float: The exponential loss of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    if set(torch.unique(true_output).numpy()) != {0, 1}:
        raise ValueError("The problem must be binary classification.")
    
    return Exponential().loss(predictions, true_output).item()


# ===============================REGRESSION===============================
def mean_squared_error(predictions, true_output):
    """
    Calculates the mean squared error of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the predictions.

    Returns:
        float: The mean squared error of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    
    return MSE().loss(predictions, true_output).item()

def root_mean_squared_error(predictions, true_output):
    """
    Calculates the root mean squared error of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the prediction.

    Returns:
        float: The root mean squared error of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    
    return sqrt(mean_squared_error(predictions, true_output))

def mean_absolute_error(predictions, true_output):
    """
    Calculates the mean absolute error of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the prediction.

    Returns:
        float: The mean absolute error of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    
    return MAE().loss(predictions, true_output).item()

def huber_loss(predictions, true_output):
    """
    Calculates the huber loss of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the prediction.

    Returns:
        float: The huber loss of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    
    return Huber().loss(predictions, true_output).item()

def r2_score(predictions, true_output):
    """
    Calculates the coefficient of determination of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the prediction.

    Returns:
        float: The coefficient of determination of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    
    residuals = true_output - predictions
    SSE = torch.sum(residuals ** 2).item()
    SST = torch.sum((true_output - torch.mean(true_output)) ** 2).item()
    r_squared = 1 - SSE / SST
    return r_squared

def adjusted_r2_score(predictions, true_output, n_features):
    """
    Calculates the adjusted coefficient of determination of the predictions.

    Args:
        predictions (torch.Tensor): A tensor of predicted as a vector. Must be the same shape as the true_output.
        true_output (torch.Tensor): A tensor of true values as a vector. Must be the same shape as the prediction.
        n_features (int): The number of features in the original data. Must be a positive integer.

    Returns:
        float: The adjusted coefficient of determination of the predictions.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(true_output, torch.Tensor):
        raise TypeError("predictions and true_output must be torch tensors.")
    if predictions.shape != true_output.shape:
        raise ValueError("predictions and true_output must have the same shape.")
    if true_output.ndim != 1:
        raise ValueError("The predictions and the true output must be a 1 dimensional tensor.")
    n_samples = len(true_output)
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer.")
    if n_samples - n_features - 1 <= 0:
        raise ValueError("number of samples must be greater than the number of features.")
    
    r_squared = r2_score(predictions, true_output)
    adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r_squared


# ===============================REGRESSION===============================
def silhouette_score(X, y, return_samples=False):
    """
    Computes the silhouette score of a clustering algorithm.

    Args:
        X (torch.Tensor): The input data of the model.
        y (torch.Tensor): The output classes.
        return_samples (bool, optional): Determines if silhouette score is returned separately or is averaged accross every sample. Defaults to False, i.e. by default returns the average value.

    Returns:
        float | torch.Tensor: if return_samples, returns a 1 dimensional torch tensor of values and if false, returns the average silhouette score.
    """
    if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("The input matrix and the label matrix must be a PyTorch tensor.")
    if X.ndim != 2:
        raise ValueError("The input matrix must be a 2 dimensional tensor.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
    vals = torch.unique(y).numpy()
    if set(vals) != {*range(len(vals))}:
        raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

    dists = torch.cdist(X, X)
    classes = torch.unique(y)
    a = torch.zeros_like(y, dtype=X.dtype)
    b = torch.zeros_like(y, dtype=X.dtype)

    for i in range(len(X)):
        mask = y == y[i]
        mask[i] = False  # remove own sample
        a[i] = torch.mean(dists[i, mask]) if torch.sum(mask) > 1 else 0
        other_distances = []
        for label in classes:
            if label != y[i]:
                mask = y == label
                other_distances.append(torch.mean(dists[i, mask]))
        if len(other_distances) > 0: b[i] = min(other_distances)
    
    s = (b - a) / torch.maximum(a, b)
    s[torch.isnan(s)] = 0

    if return_samples:
        return s
    return torch.mean(s).item()
