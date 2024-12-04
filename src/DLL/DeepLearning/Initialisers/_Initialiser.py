from abc import ABC, abstractmethod


class Initialiser(ABC):
    @abstractmethod
    def initialise(self, shape):
        pass

    def _get_dims(self, shape):
        shape = tuple(shape)
        if len(shape) < 1:
            input_dim = output_dim = 1
        elif len(shape) == 1:
            input_dim = output_dim = shape[0]
        elif len(shape) == 2:
            input_dim = shape[0]
            output_dim = shape[0]
        else:  # convolutional layers (output_depth, input_depth, ...)
            product = 1
            for dim in shape[2:]:
                product *= dim
            input_dim = product * shape[1]
            output_dim = product * shape[0]
        return int(input_dim), int(output_dim)
