import numpy as np

from tiledb.bioimg.helpers import get_decimal_from_rgba, get_rgba, iter_color


def test_color_iterator():
    generator_grayscale = iter_color(np.dtype(np.uint8), 1)
    generator_rgb = iter_color(np.dtype(np.uint8), 3)
    generator_random = iter_color(np.dtype(np.uint8), 5)

    for _, color in zip(range(1), generator_grayscale):
        assert color == get_rgba(get_decimal_from_rgba(color))
        assert color == {"red": 255, "green": 255, "blue": 255, "alpha": 255}

    for idx, color in zip(range(3), generator_rgb):
        assert color == get_rgba(get_decimal_from_rgba(color))
        assert color == {
            "red": 255 if idx == 0 else 0,
            "green": 255 if idx == 1 else 0,
            "blue": 255 if idx == 2 else 0,
            "alpha": 255,
        }

    for _, color in zip(range(5), generator_random):
        assert color == get_rgba(get_decimal_from_rgba(color))
