import torch


def accuracy(predictions, true_output):
    # multi-class accuracy
    if len(predictions.shape) == 2 and predictions.shape[1] > 1:
        prediction_indicies = torch.argmax(predictions, dim=1)
        correct_indicies = torch.argmax(true_output, dim=1)
        correct = prediction_indicies == correct_indicies
        return torch.sum(correct).item() / correct.shape[0]
    # 2-class accuracy
    else:
        prediction_values = predictions >= 0.5
        correct = prediction_values == true_output
        return torch.sum(correct).item() / correct.shape[0]
