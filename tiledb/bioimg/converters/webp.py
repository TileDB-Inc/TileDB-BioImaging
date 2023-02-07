from .axes import YXC_TO_YX, Axes, AxesMapper


def ToWebPAxesMapper(source: Axes, c_size: int) -> AxesMapper:
    """Mapper from 3D axes (YXC or permutations thereof) to 2D (YX)"""
    mapper = source.mapper(Axes("YXC"))
    mapper._transforms += (YXC_TO_YX(c_size),)
    return mapper
