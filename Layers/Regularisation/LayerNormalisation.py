from Layers.Regularisation.GroupNormalisation import GroupNorm1d


class LayerNorm1d(GroupNorm1d):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size, 1, **kwargs)
        self.name = "Layer normalisation"