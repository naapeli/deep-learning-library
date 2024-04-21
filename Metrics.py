import torch


def accuracy(predictions, true_output):
    prediction_indicies = torch.argmax(predictions, dim=1)
    correct_indicies = torch.argmax(true_output, dim=1)
    correct = prediction_indicies == correct_indicies
    return torch.sum(correct).item() / correct.shape[0]
