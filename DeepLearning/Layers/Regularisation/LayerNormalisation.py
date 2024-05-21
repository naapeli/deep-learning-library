from .GroupNormalisation import GroupNorm1d


class LayerNorm1d(GroupNorm1d):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, 1, **kwargs)
        self.name = "Layer normalisation"
