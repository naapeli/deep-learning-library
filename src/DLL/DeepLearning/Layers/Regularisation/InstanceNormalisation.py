from .GroupNormalisation import GroupNorm


class InstanceNorm(GroupNorm):
    def __init__(self, output_shape=None, **kwargs):
        if output_shape: super().__init__(output_shape, output_shape[1], **kwargs)
        self.name = "Instance normalisation"

    def initialise_layer(self, input_shape, data_type, device):
        self.num_groups = input_shape[1]
        super().initialise_layer(input_shape, data_type, device)
        