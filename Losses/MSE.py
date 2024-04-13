import numpy as np


class mse:
    def loss(prediction, true_output):
        return np.mean((prediction - true_output) ** 2)

    def gradient(prediction, true_output):
        return 2 * (prediction - true_output) / np.size(prediction)
