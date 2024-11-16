import torch


class DataReader:
    """
    The data reader.

    Args:
        X (torch.Tensor of shape (n_samples, ...)): The input data.
        Y (torch.Tensor of shape (n_samples, ...)): The target labels or values.
        batch_size (int, optional): The batch size. If the batch size is larger than the number of samples, the maximum number of samples is used as the batch_size. Must be a positive integer. Defaults to 64.
        shuffle (bool, optional): Determines if the input data is shuffled in the beginning. Defaults to True.
        shuffle_every_epoch (bool, optional): Determines if the input data is shuffled every time all data points are used. Defaults to False.
    """
    def __init__(self, X, Y, batch_size=64, shuffle=True, shuffle_every_epoch=False):
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor.")
        if not isinstance(Y, torch.Tensor):
            raise TypeError("Y must be a torch.Tensor.")
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of samples.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive integer.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        if not isinstance(shuffle_every_epoch, bool):
            raise TypeError("shuffle_every_epoch must be a boolean.")

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
        self.batch_size = batch_size if len(Y) >= batch_size else len(Y)

    def get_data(self):
        """
        A generator giving going through the entire dataset.

        Yields:
            tuple[torch.Tensor, torch.Tensor]: X_batch, y_batch
        """
        iteration = 0
        while iteration * self.batch_size < self.data_length:
            yield self.X[iteration * self.batch_size:(iteration + 1) * self.batch_size], self.Y[iteration * self.batch_size:(iteration + 1) * self.batch_size]
            iteration += 1
        if self.shuffle_every_epoch and self.shuffle:
            self.perm = torch.randperm(self.data_length, device=self.X.device)
            self.X = self.X.index_select(0, self.perm)
            self.Y = self.Y.index_select(0, self.perm)
