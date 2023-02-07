from .axes import YXC_TO_YX, Axes, AxesMapper


class ToWebPAxesMapper(AxesMapper):
    """Mapper from 3D axes (YXC or permutations thereof) to 2D (YX)"""

    def __init__(self, source: Axes, c_size: int):
        super().__init__(source, target=Axes("YXC"))
        self._transforms.append(YXC_TO_YX(c_size))
