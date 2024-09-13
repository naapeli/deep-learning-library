from .GroupNormalisation import GroupNorm


class LayerNorm(GroupNorm):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, 1, **kwargs)
        self.name = "Layer normalisation"
