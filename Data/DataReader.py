import torch


class DataReader:
    def __init__(self, X, Y, batch_size=64, shuffle=True, new_shuffle_per_epoch=False, **kwargs):
        self.axis = 0
        self.data_length = Y.size(self.axis)
        self.shuffle = shuffle
        self.new_shuffle_per_epoch = new_shuffle_per_epoch
        if self.shuffle:
            self.perm = torch.randperm(self.data_length, requires_grad=False)
            self.X = X.index_select(self.axis, self.perm)
            self.Y = Y.index_select(self.axis, self.perm)
        else:
            self.X = X
            self.Y = Y
        self.batch_size = batch_size

    def get_data(self):
        iteration = 0
        while iteration * self.batch_size < self.data_length:
            yield self.X[iteration * self.batch_size:(iteration + 1) * self.batch_size], self.Y[iteration * self.batch_size:(iteration + 1) * self.batch_size]
            iteration += 1
        if self.new_shuffle_per_epoch and self.shuffle:
            self.perm = torch.randperm(self.data_length, requires_grad=False)
            self.X = self.X.index_select(self.axis, self.perm)
            self.Y = self.Y.index_select(self.axis, self.perm)
