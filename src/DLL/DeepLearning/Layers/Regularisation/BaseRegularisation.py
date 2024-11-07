from ..Activations.Activation import Activation
from ....Exceptions import NotCompiledError


class BaseRegularisation(Activation):
    def summary(self):
        if not hasattr(self, "input_shape"):
            raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        params_summary = " - Parameters: " + str(self.nparams) if self.nparams > 0 else ""
        return f"{self.name} - Output: ({output_shape})" + params_summary
