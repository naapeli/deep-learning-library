from Layers.Regularisation.GroupNormalisation import GroupNorm1d


class InstanceNorm1d(GroupNorm1d):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size, output_size, **kwargs)
        self.name = "Instance normalisation"

    def set_output_size(self, output_size):
        self.num_groups = output_size
        super().set_output_size(output_size)
        