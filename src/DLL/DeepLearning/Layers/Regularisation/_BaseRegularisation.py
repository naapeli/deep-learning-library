from ..Activations._Activation import Activation
from ....Exceptions import NotCompiledError


class BaseRegularisation(Activation):
    def summary(self, offset=""):
        if not hasattr(self, "input_shape"):
            raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        params_summary = " - Parameters: " + str(self.nparams) if self.nparams > 0 else ""
        return offset + f"{self.name} - Output: ({output_shape})" + params_summary
