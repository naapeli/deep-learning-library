from . import Callback
from ...Exceptions import NotCompiledError


class EarlyStopping(Callback):
    """
    The early stopping callback. Stops the training of the model early if no improvement is made.

    Args:
        monitor (str, optional): The quantity, which is monitored for the early stop. Must be in the metrics of the model. Defaults to "val_loss".
        patience (int, optional): The number of epochs the model waits without improvement. Must be a positive integer. Defaults to 1.
        mode (str, optional): Determines if the monitored metric better if it is maximized or minimized. Must be in ["min", "max"]. Defaults to "min".
        restore_best_model (bool, optional): Determines if the best model is restored after the training is finished. Defaults to False.
        warmup_length (int, optional): Determines how many epochs are trained no matter what. Must be non-negative. Defaults to 0.
        verbose (bool, optional): Determines if information about the callbacks is printed. Must be a boolean. Defaults to False.
    Note:
        Using restore_best_model = True considerably slows down training as the best model must be saved every epoch the model improves.
    """
    def __init__(self, monitor="val_loss", patience=1, mode="min", restore_best_model=False, warmup_length=0, verbose=False):
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if mode not in ["min", "max"]:
            raise ValueError('mode must be in ["min", "max"].')
        if not isinstance(restore_best_model, bool):
            raise TypeError("restore_best_model must be a boolean.")
        if not isinstance(warmup_length, int) or warmup_length < 0:
            raise ValueError("warmup_length must be non-negative.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")

        self.monitor = monitor
        self.patience = patience
        self.mode = 1 if mode == "min" else -1
        self.restore_best_model = restore_best_model
        self.warmup_length = warmup_length
        self.verbose = verbose

    def set_model(self, model):
        """
        Lets the callback know about the chosen model. Is automatically called when using Model.fit()

        Args:
            Model (:ref:`models_section_label`): The chosen model.
        """
        if not hasattr(model, "metrics") or self.monitor not in model.metrics:
            raise ValueError("monitor must be in the metrics of the model.")
        
        self.model = model

    def on_train_start(self):
        """
        Sets the needed attributes.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")

        if self.restore_best_model:
            self.stored_model = self.model.clone()
        self.wait = 0
        self.best_value = float("inf")

    def on_train_end(self):
        """
        Resets the model to the stored model if restore_best_model was chosen.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")
        
        if self.restore_best_model:
            self.model.layers = self.stored_model.layers

    def on_epoch_end(self, epoch, metrics):
        """
        Calculates if the model has improved on the last epoch.

        Args:
            epoch (int): The current epoch.
            metrics (dict[str, float]): The values of the chosen metrics of the last epoch.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")

        value = metrics[self.monitor]
        if self.mode * value < self.best_value:
            self.wait = 0
            self.best_value = self.mode * value
            if self.restore_best_model:
                self.stored_model = self.model.clone()
        elif epoch > self.warmup_length:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose: print(f"Stopped training early at epoch {epoch + 1} after {self.patience} epochs of no improvement. The monitored metric is currently {value:.3f} and the best model has the value of {self.mode * self.best_value:.3f}.")
                self.model.stop_training = True
