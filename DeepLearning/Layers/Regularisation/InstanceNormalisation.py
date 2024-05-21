from .GroupNormalisation import GroupNorm1d


class InstanceNorm1d(GroupNorm1d):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, output_shape[1], **kwargs)
        self.name = "Instance normalisation"

    def set_output_shape(self, output_shape):
        self.num_groups = output_shape[1]
        super().set_output_shape(output_shape)
        