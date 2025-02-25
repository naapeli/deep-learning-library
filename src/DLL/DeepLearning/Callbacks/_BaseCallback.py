from abc import ABC


class Callback(ABC):
    def set_model(self, model):
        """
        Lets the callback know about the chosen model. Is automatically called when using Model.train()

        Args:
            Model (:ref:`models_section_label`): The chosen model.
        """
        self.model = model

    def on_train_start(self):
        """
        A method, which is automatically called before the training starts.
        """
        pass

    def on_train_end(self):
        """
        A method, which is automatically called after training.
        """
        pass

    def on_epoch_end(self, epoch, metrics):
        """
        A method, which is automatically called after every epoch.

        Args:
            epoch (int): The current epoch.
            metrics (dict[str, float]): The values of the chosen metrics of the last epoch.
        """
        pass

    def on_batch_end(self, epoch):
        """
        A method, which is automatically called after every batch.

        Args:
            epoch (int): The current epoch.
        """
        pass
