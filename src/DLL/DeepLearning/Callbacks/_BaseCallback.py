import torch

from abc import ABC


class Callback(ABC):
    def set_model(self, model):
        self.model = model

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_end(self, epoch):
        pass
