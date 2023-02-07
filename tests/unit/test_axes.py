import itertools as it

import numpy as np
import pytest

from tiledb.bioimg.converters.axes import Axes, Move, Squeeze, Swap, Unsqueeze


class TestAxesMappers:
    @pytest.mark.parametrize(
        "s,i,j", [(b"ADCBE", 1, 3), (b"DBCAE", 3, 0), (b"ACBDE", 2, 1)]
    )
    def test_swap(self, s, i, j):
        axes_mapper = Swap(i, j)
        b = bytearray(s)
        assert axes_mapper.transform_sequence(b) is None
        assert b == b"ABCDE"

    def test_swap_array(self):
        axes_mapper = Swap(1, 3)
        a = np.empty((5, 4, 8, 3, 6))
        np.testing.assert_array_equal(axes_mapper.map_array(a), np.swapaxes(a, 1, 3))

    @pytest.mark.parametrize(
        "s,i,j",
        [
            (b"ADBCE", 1, 3),
            (b"ACDBE", 3, 1),
            (b"ACBDE", 1, 2),
            (b"ACBDE", 2, 1),
            (b"EABCD", 0, 4),
            (b"BCDEA", 4, 0),
        ],
    )
    def test_move(self, s, i, j):
        axes_mapper = Move(i, j)
        b = bytearray(s)
        assert axes_mapper.transform_sequence(b) is None
        assert b == b"ABCDE"

    def test_move_array(self):
        axes_mapper = Move(1, 3)
        a = np.empty((5, 4, 8, 3, 6))
        np.testing.assert_array_equal(axes_mapper.map_array(a), np.moveaxis(a, 1, 3))

    @pytest.mark.parametrize(
        "s,idxs",
        [
            (b"ADBC", (1,)),
            (b"ADBCF", (1, 4)),
            (b"ADBCEF", (1, 4, 5)),
            (b"DAEBFC", (0, 2, 4)),
        ],
    )
    def test_squeeze(self, s, idxs):
        axes_mapper = Squeeze(idxs)
        b = bytearray(s)
        assert axes_mapper.transform_sequence(b) is None
        assert b == b"ABC"

    def test_squeeze_array(self):
        axes_mapper = Squeeze((1, 3))
        a = np.empty((5, 1, 8, 1, 6))
        np.testing.assert_array_equal(axes_mapper.map_array(a), np.squeeze(a, (1, 3)))

    @pytest.mark.parametrize(
        "s,idxs,t",
        [
            (b"ABC", (1,), b"A_BC"),
            (b"ABC", (1, 2), b"A__BC"),
            (b"ABC", (1, 3), b"A_B_C"),
            (b"ABC", (1, 3, 4), b"A_B__C"),
            (b"ABC", (1, 3, 5), b"A_B_C_"),
        ],
    )
    def test_unsqueeze(self, s, idxs, t):
        axes_mapper = Unsqueeze(idxs)
        b = bytearray(s)
        assert axes_mapper.transform_sequence(b, fill_value=ord("_")) is None
        assert b == t

    def test_unsqueeze_array(self):
        axes_mapper = Unsqueeze((1, 3))
        a = np.empty((5, 8, 6))
        np.testing.assert_array_equal(
            axes_mapper.map_array(a), np.expand_dims(a, (1, 3))
        )


class TestAxes:
    def test_init(self):
        assert Axes("XYZ").dims == "XYZ"

        with pytest.raises(ValueError) as excinfo:
            Axes("XYZW")
        assert str(excinfo.value) == "'W' is not a valid Axis"

        with pytest.raises(ValueError) as excinfo:
            Axes("XYZX")
        assert "Duplicate axes" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Axes("ZYTC")
        assert str(excinfo.value) == "Missing required axis 'X'"

        with pytest.raises(ValueError) as excinfo:
            Axes("XTC")
        assert str(excinfo.value) == "Missing required axis 'Y'"

    @pytest.mark.parametrize(
        "canonical_dims",
        ["YX", "ZYX", "CYX", "TYX", "CZYX", "TCYX", "TZYX", "TCZYX"],
    )
    def test_canonical_unsqueezed(self, canonical_dims):
        shape = np.random.randint(2, 20, size=len(canonical_dims))
        for axes in map(Axes, it.permutations(canonical_dims)):
            assert axes.canonical(shape).dims == canonical_dims

    def test_canonical_squeezed(self):
        shape = (1, 60, 40)
        for s in "ZXY", "CXY", "TXY":
            assert Axes(s).canonical(shape) == Axes("YX")

        shape = (1, 1, 60, 40)
        for s in "CZXY", "TCXY", "TZXY":
            assert Axes(s).canonical(shape) == Axes("YX")

        shape = (3, 1, 60, 40)
        assert Axes("CZXY").canonical(shape) == Axes("CYX")
        assert Axes("TCXY").canonical(shape) == Axes("TYX")
        assert Axes("ZTXY").canonical(shape) == Axes("ZYX")

        shape = (1, 1, 1, 60, 40)
        for s in "TCZXY", "TZCXY", "CZTXY":
            assert Axes(s).canonical(shape) == Axes("YX")

        shape = (1, 3, 1, 60, 40)
        assert Axes("TCZXY").canonical(shape) == Axes("CYX")
        assert Axes("ZTCXY").canonical(shape) == Axes("TYX")
        assert Axes("CZTXY").canonical(shape) == Axes("ZYX")

        shape = (7, 3, 1, 60, 40)
        assert Axes("CTZXY").canonical(shape) == Axes("TCYX")
        assert Axes("ZTCXY").canonical(shape) == Axes("TZYX")
        assert Axes("ZCTXY").canonical(shape) == Axes("CZYX")


class TestCompositeAxesMapper:
    def test_canonical_transform_2d(self):
        a = np.random.rand(60, 40)
        assert_canonical_transform("YX", a, a)
        assert_canonical_transform("XY", a, np.swapaxes(a, 0, 1))

    def test_canonical_transform_3d(self):
        a = np.random.rand(10, 60, 40)
        for s in "ZYX", "CYX", "TYX":
            assert_canonical_transform(s, a, a)
        for s in "XYZ", "XYC", "XYT":
            assert_canonical_transform(s, a, np.swapaxes(a, 0, 2))
        for s in "ZXY", "CXY", "TXY":
            assert_canonical_transform(s, a, np.swapaxes(a, 1, 2))
        for s in "YXZ", "YXC", "YXT":
            assert_canonical_transform(s, a, np.moveaxis(a, 2, 0))

    def test_canonical_transform_4d(self):
        a = np.random.rand(3, 10, 60, 40)
        for s in "CZYX", "TCYX", "TZYX":
            assert_canonical_transform(s, a, a)
        for s in "ZCYX", "CTYX", "ZTYX":
            assert_canonical_transform(s, a, np.swapaxes(a, 0, 1))
        for s in "CZXY", "TCXY", "TZXY":
            assert_canonical_transform(s, a, np.swapaxes(a, 2, 3))
        for s in "ZYXC", "CYXT", "ZYXT":
            assert_canonical_transform(s, a, np.moveaxis(a, 3, 0))
        for s in "CYXZ", "TYXC", "TYXZ":
            assert_canonical_transform(s, a, np.moveaxis(a, 3, 1))
        for s in "YXZC", "YXCT", "YXZT":
            assert_canonical_transform(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 0))
        for s in "YXCZ", "YXTC", "YXTZ":
            assert_canonical_transform(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 1))
        for s in "ZCXY", "CTXY", "ZTXY":
            assert_canonical_transform(s, a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 3))
        for s in "XYZC", "XYCT", "XYZT":
            assert_canonical_transform(s, a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 2))
        for s in "CXYZ", "TXYC", "TXYZ":
            assert_canonical_transform(s, a, np.swapaxes(np.moveaxis(a, 3, 1), 2, 3))
        for s in "ZXYC", "CXYT", "ZXYT":
            assert_canonical_transform(s, a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 3))
        for s in "XYCZ", "XYTC", "XYTZ":
            assert_canonical_transform(s, a, np.swapaxes(np.moveaxis(a, 2, 0), 1, 3))

    def test_canonical_transform_5d(self):
        a = np.random.rand(7, 3, 10, 60, 40)
        assert_canonical_transform("TCZYX", a, a)

        assert_canonical_transform("CTZYX", a, np.swapaxes(a, 0, 1))
        assert_canonical_transform("ZCTYX", a, np.swapaxes(a, 0, 2))
        assert_canonical_transform("TZCYX", a, np.swapaxes(a, 1, 2))
        assert_canonical_transform("TCXYZ", a, np.swapaxes(a, 2, 4))
        assert_canonical_transform("TCZXY", a, np.swapaxes(a, 3, 4))

        assert_canonical_transform("ZTCYX", a, np.moveaxis(a, 0, 2))
        assert_canonical_transform("CZTYX", a, np.moveaxis(a, 2, 0))
        assert_canonical_transform("CZYXT", a, np.moveaxis(a, 4, 0))
        assert_canonical_transform("TZYXC", a, np.moveaxis(a, 4, 1))
        assert_canonical_transform("TCYXZ", a, np.moveaxis(a, 4, 2))

        assert_canonical_transform("CTXYZ", a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 4))
        assert_canonical_transform("CTZXY", a, np.swapaxes(np.swapaxes(a, 0, 1), 3, 4))
        assert_canonical_transform("ZCTXY", a, np.swapaxes(np.swapaxes(a, 0, 2), 3, 4))
        assert_canonical_transform("YXZTC", a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 4))
        assert_canonical_transform("XYZCT", a, np.swapaxes(np.swapaxes(a, 0, 4), 1, 3))
        assert_canonical_transform("ZCXYT", a, np.swapaxes(np.swapaxes(a, 0, 4), 2, 4))
        assert_canonical_transform("TZCXY", a, np.swapaxes(np.swapaxes(a, 1, 2), 3, 4))
        assert_canonical_transform("TZXYC", a, np.swapaxes(np.swapaxes(a, 1, 4), 2, 4))

        assert_canonical_transform("ZTXYC", a, np.swapaxes(np.moveaxis(a, 0, 2), 1, 4))
        assert_canonical_transform("ZTCXY", a, np.swapaxes(np.moveaxis(a, 0, 2), 3, 4))
        assert_canonical_transform("YXCZT", a, np.swapaxes(np.moveaxis(a, 0, 3), 0, 4))
        assert_canonical_transform("XYCZT", a, np.swapaxes(np.moveaxis(a, 1, 3), 0, 4))
        assert_canonical_transform("CZTXY", a, np.swapaxes(np.moveaxis(a, 2, 0), 3, 4))
        assert_canonical_transform("CXYTZ", a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 4))
        assert_canonical_transform("CXYZT", a, np.swapaxes(np.moveaxis(a, 4, 0), 2, 4))
        assert_canonical_transform("CZXYT", a, np.swapaxes(np.moveaxis(a, 4, 0), 3, 4))
        assert_canonical_transform("ZTYXC", a, np.swapaxes(np.moveaxis(a, 4, 1), 0, 2))
        assert_canonical_transform("TXYZC", a, np.swapaxes(np.moveaxis(a, 4, 1), 2, 4))
        assert_canonical_transform("CTYXZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 0, 1))
        assert_canonical_transform("TXYCZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 1, 4))

        assert_canonical_transform("XYTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 3))
        assert_canonical_transform("YXTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 4))
        assert_canonical_transform("TYXCZ", a, np.moveaxis(np.moveaxis(a, 1, 4), 1, 4))
        assert_canonical_transform("ZYXTC", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 1))
        assert_canonical_transform("CYXTZ", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 2))
        assert_canonical_transform("TYXZC", a, np.moveaxis(np.moveaxis(a, 4, 1), 4, 2))
        assert_canonical_transform("ZCYXT", a, np.moveaxis(np.moveaxis(a, 4, 0), 1, 2))
        assert_canonical_transform("ZYXCT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 1))
        assert_canonical_transform("CYXZT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 2))

    def test_transform_squeeze(self):
        a = np.random.rand(1, 60, 40)
        assert_transform("ZXY", "XY", a, np.squeeze(a, 0))
        assert_transform("ZXY", "YX", a, np.swapaxes(np.squeeze(a, 0), 0, 1))

        a = np.random.rand(1, 1, 60, 40)
        assert_transform("CZXY", "XY", a, np.squeeze(a, (0, 1)))
        assert_transform("CZXY", "CXY", a, np.squeeze(a, 1))
        assert_transform("CZXY", "ZXY", a, np.squeeze(a, 0))
        assert_transform("CZXY", "YX", a, np.swapaxes(np.squeeze(a, (0, 1)), 0, 1))
        assert_transform("CZXY", "CYX", a, np.swapaxes(np.squeeze(a, 1), 1, 2))
        assert_transform("CZXY", "ZYX", a, np.swapaxes(np.squeeze(a, 0), 1, 2))

        a = np.random.rand(1, 1, 1, 60, 40)
        assert_transform("TCZXY", "XY", a, np.squeeze(a, (0, 1, 2)))
        assert_transform("TCZXY", "ZXY", a, np.squeeze(a, (0, 1)))
        assert_transform("TCZXY", "CXY", a, np.squeeze(a, (0, 2)))
        assert_transform("TCZXY", "TXY", a, np.squeeze(a, (1, 2)))
        assert_transform("TCZXY", "CZXY", a, np.squeeze(a, 0))
        assert_transform("TCZXY", "TZXY", a, np.squeeze(a, 1))
        assert_transform("TCZXY", "TCXY", a, np.squeeze(a, 2))
        assert_transform("TCZXY", "YX", a, np.swapaxes(np.squeeze(a, (0, 1, 2)), 0, 1))
        assert_transform("TCZXY", "ZYX", a, np.swapaxes(np.squeeze(a, (0, 1)), 1, 2))
        assert_transform("TCZXY", "CYX", a, np.swapaxes(np.squeeze(a, (0, 2)), 1, 2))
        assert_transform("TCZXY", "TYX", a, np.swapaxes(np.squeeze(a, (1, 2)), 1, 2))
        assert_transform("TCZXY", "CZYX", a, np.swapaxes(np.squeeze(a, 0), 2, 3))
        assert_transform("TCZXY", "TZYX", a, np.swapaxes(np.squeeze(a, 1), 2, 3))
        assert_transform("TCZXY", "TCYX", a, np.swapaxes(np.squeeze(a, 2), 2, 3))

    def test_transform_expand(self):
        a = np.random.rand(10, 60, 40)
        assert_transform("CYX", "ZCYX", a, np.expand_dims(a, 0))
        assert_transform("CYX", "CZYX", a, np.expand_dims(a, 1))
        assert_transform("CYX", "TZCYX", a, np.expand_dims(a, (0, 1)))
        assert_transform("CYX", "TCZYX", a, np.expand_dims(a, (0, 2)))
        assert_transform("CYX", "CTZYX", a, np.expand_dims(a, (1, 2)))
        assert_transform("CYX", "ZCXY", a, np.swapaxes(np.expand_dims(a, 0), 2, 3))
        assert_transform("CYX", "CZXY", a, np.swapaxes(np.expand_dims(a, 1), 2, 3))
        assert_transform(
            "CYX", "TZCXY", a, np.swapaxes(np.expand_dims(a, (0, 1)), 3, 4)
        )
        assert_transform(
            "CYX", "TCZXY", a, np.swapaxes(np.expand_dims(a, (0, 2)), 3, 4)
        )
        assert_transform(
            "CYX", "CTZXY", a, np.swapaxes(np.expand_dims(a, (1, 2)), 3, 4)
        )


def assert_transform(source, target, a, expected):
    axes_mapper = Axes(source).mapper(Axes(target))
    assert axes_mapper.map_shape(a.shape) == expected.shape
    np.testing.assert_array_equal(axes_mapper.map_array(a), expected)


def assert_canonical_transform(source, a, expected):
    source = Axes(source)
    target = source.canonical(a.shape)
    axes_mapper = source.mapper(target)
    assert axes_mapper.map_shape(a.shape) == expected.shape
    np.testing.assert_array_equal(axes_mapper.map_array(a), expected)
