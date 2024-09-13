import torch


class DataReader:
    def __init__(self, X, Y, batch_size=64, shuffle=True, shuffle_every_epoch=False, **kwargs):
        self.data_length = Y.size(0)
        self.shuffle = shuffle
        self.shuffle_every_epoch = shuffle_every_epoch
        if self.shuffle:
            self.perm = torch.randperm(self.data_length, device=X.device)
            self.X = X.index_select(0, self.perm)
            self.Y = Y.index_select(0, self.perm)
        else:
            self.X = X
            self.Y = Y
        self.batch_size = batch_size

    def get_data(self):
        iteration = 0
        while iteration * self.batch_size < self.data_length:
            yield self.X[iteration * self.batch_size:(iteration + 1) * self.batch_size], self.Y[iteration * self.batch_size:(iteration + 1) * self.batch_size]
            iteration += 1
        if self.shuffle_every_epoch and self.shuffle:
            self.perm = torch.randperm(self.data_length, device=self.X.device)
            self.X = self.X.index_select(0, self.perm)
            self.Y = self.Y.index_select(0, self.perm)
