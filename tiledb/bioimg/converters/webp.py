from .axes import YX_TO_YXC, YXC_TO_YX, Axes, AxesMapper


class ToWebPAxesMapper(AxesMapper):
    """Mapper from 3D axes (YXC or permutations thereof) to 2D (YX)"""

    def __init__(self, source: Axes, c_size: int):
        super().__init__(source, target=Axes("YXC"))
        self._transforms.append(YXC_TO_YX(c_size))


class FromWebPAxesMapper(AxesMapper):
    """Mapper from 2D axes (YX) to 3D (YXC or permutations thereof)"""

    def __init__(self, target: Axes, c_size: int):
        super().__init__(Axes("YXC"), target)
        self._transforms.insert(0, YX_TO_YXC(c_size))
