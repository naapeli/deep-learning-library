class NotFittedError(Exception):
    """
    Exception raised when an action is called before the model is fitted.

    Args:
        message (string): The message which is passed to the Exception class.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
