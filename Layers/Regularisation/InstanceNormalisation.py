from Layers.Regularisation.GroupNormalisation import GroupNorm1d


class InstanceNorm1d(GroupNorm1d):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, output_shape, **kwargs)
        self.name = "Instance normalisation"

    def set_output_shape(self, output_shape):
        self.num_groups = output_shape
        super().set_output_shape(output_shape)
        