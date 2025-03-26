from warnings import warn

from . import Callback
from ...Exceptions import NotCompiledError


class ReduceLROnPlateau(Callback):
    """
    The reduce learning rate on plateau callback. Reduces the learning rate of the model optimiser if no improvement is made.

    Args:
        monitor (str, optional): The quantity, which is monitored. Must be in the metrics of the model. Defaults to "val_loss".
        patience (int, optional): The number of epochs the model waits without improvement. Must be a positive integer. Defaults to 0.
        mode (str, optional): Determines if the monitored metric better if it is maximized or minimized. Must be in ["min", "max"]. Defaults to "min".
        warmup_length (int, optional): Determines how many epochs are trained on the original learning rate. Must be non-negative. Defaults to 0.
        factor (float, optional): The learning rate is multiplied by this factor when training. Must be in range (0, 1). Defaults to 0.5.
        verbose (bool, optional): Determines if information about the callbacks is printed. Must be a boolean. Defaults to False.
    """
    def __init__(self, monitor="val_loss", patience=0, mode="min", warmup_length=0, factor=0.5, verbose=False):
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if mode not in ["min", "max"]:
            raise ValueError('mode must be in ["min", "max"].')
        if not isinstance(warmup_length, int) or warmup_length < 0:
            raise ValueError("warmup_length must be non-negative.")
        if not isinstance(factor, float) or factor <= 0 or factor >= 1:
            raise ValueError("factor must be in range (0, 1).")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")

        self.monitor = monitor
        self.patience = patience
        self.mode = 1 if mode == "min" else -1
        self.warmup_length = warmup_length
        self.factor = factor
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
        Initializes the needed attributes.

        Raises:
            NotCompiledError: callback.set_model must be called before the training starts
        """
        if not hasattr(self, "model"):
            raise NotCompiledError("callback.set_model must be called before the training starts.")
        if not hasattr(self.model.optimiser, "learning_rate"):
            warn("The chosen optimizer has no learning_rate attribute. The ReduceLROnPlateau callback has no effect.")

        self.wait = 0
        self.best_value = float("inf")

    def on_epoch_end(self, epoch, metrics):
        """
        Calculates if the learning rate should be removed.

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
        elif epoch > self.warmup_length:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr_message = None
                if hasattr(self.model.optimiser, "learning_rate"):
                    self.wait = 0
                    self.model.optimiser.learning_rate *= self.factor
                    new_lr_message = self.model.optimiser.learning_rate
                    if self.verbose: print(f"Learning rate reduced at epoch {epoch + 1} after {self.patience} epochs of no improvement. The monitored metric is currently {value:.3f} and the best model has the value of {self.mode * self.best_value:.3f}. The current learning rate is {new_lr_message}.")
