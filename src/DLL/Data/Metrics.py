import torch


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
