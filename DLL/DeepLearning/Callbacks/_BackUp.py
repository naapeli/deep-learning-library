from . import Callback
from ...Exceptions import NotCompiledError
from ..Model import save_model


class BackUp(Callback):
    """
    Saves the fitted model to the spesified filepath.

    Args:
        filepath (str | , optional): The filepath where the model is saved. Defaults to "./model.pkl".
        frequency (int, optional): The frequency of saving the model. Must be a positive integer. Defaults to 1.
        on_batch (bool, optional): Determines if a model is saved every frequency batch or epoch. Defaults to False.
        verbose (bool, optional): Determines if information about the callbacks is printed. Must be a boolean. Defaults to False.
    """
    def __init__(self, filepath="./model.pkl", frequency=1, on_batch=False, verbose=False):
        if not isinstance(filepath, str):
            raise TypeError("filepath must be a string.")
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError("frequency must be a positive integer.")
        if not isinstance(on_batch, bool):
            raise TypeError("on_batch must be boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")

        self.filepath = filepath
        self.frequency = frequency
        self.on_batch = on_batch
        self.counter = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, metrics):
        """
        Calculates if the model should be backed up on this epoch.

        Args:
            epoch (int): The current epoch.
            metrics (dict[str, float]): The values of the chosen metrics of the last epoch.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")
        
        if not self.on_batch:
            self.counter += 1
            if self.counter % self.frequency == 0:
                save_model(self.model, filepath=self.filepath)
                if self.verbose: print(f"Model backed up at epoch {epoch + 1} at {self.filepath}")
        else:
            self.counter = 0

    def on_batch_end(self, epoch):
        """
        Calculates if the model should be backed up on this batch.

        Args:
            epoch (int): The current epoch.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")
    
        if self.on_batch:
            self.counter += 1
            if self.counter % self.frequency == 0:
                save_model(self.model, filepath=self.filepath)
                if self.verbose: print(f"Model backed up at epoch {epoch + 1} at {self.filepath}")
