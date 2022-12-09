import itertools as it

import numpy as np
import pytest

from tiledb.bioimg.converters.axes import (Axes, Move, Swap,
                                           minimize_transpositions,
                                           transpose_array)


class TestTranspositions:
    @pytest.mark.parametrize(
        "s,i,j", [(b"ADCBE", 1, 3), (b"DBCAE", 3, 0), (b"ACBDE", 2, 1)]
    )
    def test_swap(self, s, i, j):
        swap = Swap(i, j)
        assert swap.transposed(s) == b"ABCDE"
        b = bytearray(s)
        assert swap.transpose(b) is None
        assert b == b"ABCDE"

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
        move = Move(i, j)
        assert move.transposed(s) == b"ABCDE"
        b = bytearray(s)
        assert move.transpose(b) is None
        assert b == b"ABCDE"

    @pytest.mark.parametrize(
        "s,t,transpositions",
        [
            ("EABCD", "ABCDE", [Move(0, 4)]),
            ("EADCB", "ABCDE", [Move(0, 4), Swap(1, 3)]),
            ("ECABD", "ABCDE", [Move(0, 4), Move(0, 2)]),
            ("AFDEBGC", "ABCDEFG", [Swap(1, 4), Move(6, 2)]),
        ],
    )
    def test_minimize_transpositions(self, s, t, transpositions):
        assert minimize_transpositions(s, t) == transpositions
        b = bytearray(s.encode())
        for transposition in transpositions:
            transposition.transpose(b)
        assert b.decode() == t


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

    def test_canonical_transpose_2d(self):
        a = np.random.rand(60, 40)
        assert_canonical_transpose("YX", a, a)
        assert_canonical_transpose("XY", a, np.swapaxes(a, 0, 1))

    def test_canonical_transpose_3d(self):
        a = np.random.rand(10, 60, 40)
        for s in "ZYX", "CYX", "TYX":
            assert_canonical_transpose(s, a, a)
        for s in "XYZ", "XYC", "XYT":
            assert_canonical_transpose(s, a, np.swapaxes(a, 0, 2))
        for s in "ZXY", "CXY", "TXY":
            assert_canonical_transpose(s, a, np.swapaxes(a, 1, 2))
        for s in "YXZ", "YXC", "YXT":
            assert_canonical_transpose(s, a, np.moveaxis(a, 2, 0))

    def test_canonical_transpose_4d(self):
        a = np.random.rand(3, 10, 60, 40)
        for s in "CZYX", "TCYX", "TZYX":
            assert_canonical_transpose(s, a, a)
        for s in "ZCYX", "CTYX", "ZTYX":
            assert_canonical_transpose(s, a, np.swapaxes(a, 0, 1))
        for s in "CZXY", "TCXY", "TZXY":
            assert_canonical_transpose(s, a, np.swapaxes(a, 2, 3))
        for s in "ZYXC", "CYXT", "ZYXT":
            assert_canonical_transpose(s, a, np.moveaxis(a, 3, 0))
        for s in "CYXZ", "TYXC", "TYXZ":
            assert_canonical_transpose(s, a, np.moveaxis(a, 3, 1))
        for s in "YXZC", "YXCT", "YXZT":
            assert_canonical_transpose(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 0))
        for s in "YXCZ", "YXTC", "YXTZ":
            assert_canonical_transpose(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 1))
        for s in "ZCXY", "CTXY", "ZTXY":
            assert_canonical_transpose(s, a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 3))
        for s in "XYZC", "XYCT", "XYZT":
            assert_canonical_transpose(s, a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 2))
        for s in "CXYZ", "TXYC", "TXYZ":
            assert_canonical_transpose(s, a, np.swapaxes(np.moveaxis(a, 3, 1), 2, 3))
        for s in "ZXYC", "CXYT", "ZXYT":
            assert_canonical_transpose(s, a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 3))
        for s in "XYCZ", "XYTC", "XYTZ":
            assert_canonical_transpose(s, a, np.swapaxes(np.moveaxis(a, 2, 0), 1, 3))

    def test_canonical_transpose_5d(self):
        a = np.random.rand(7, 3, 10, 60, 40)
        assert_canonical_transpose("TCZYX", a, a)

        assert_canonical_transpose("CTZYX", a, np.swapaxes(a, 0, 1))
        assert_canonical_transpose("ZCTYX", a, np.swapaxes(a, 0, 2))
        assert_canonical_transpose("TZCYX", a, np.swapaxes(a, 1, 2))
        assert_canonical_transpose("TCXYZ", a, np.swapaxes(a, 2, 4))
        assert_canonical_transpose("TCZXY", a, np.swapaxes(a, 3, 4))

        assert_canonical_transpose("ZTCYX", a, np.moveaxis(a, 0, 2))
        assert_canonical_transpose("CZTYX", a, np.moveaxis(a, 2, 0))
        assert_canonical_transpose("CZYXT", a, np.moveaxis(a, 4, 0))
        assert_canonical_transpose("TZYXC", a, np.moveaxis(a, 4, 1))
        assert_canonical_transpose("TCYXZ", a, np.moveaxis(a, 4, 2))

        assert_canonical_transpose("CTXYZ", a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 4))
        assert_canonical_transpose("CTZXY", a, np.swapaxes(np.swapaxes(a, 0, 1), 3, 4))
        assert_canonical_transpose("ZCTXY", a, np.swapaxes(np.swapaxes(a, 0, 2), 3, 4))
        assert_canonical_transpose("YXZTC", a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 4))
        assert_canonical_transpose("XYZCT", a, np.swapaxes(np.swapaxes(a, 0, 4), 1, 3))
        assert_canonical_transpose("ZCXYT", a, np.swapaxes(np.swapaxes(a, 0, 4), 2, 4))
        assert_canonical_transpose("TZCXY", a, np.swapaxes(np.swapaxes(a, 1, 2), 3, 4))
        assert_canonical_transpose("TZXYC", a, np.swapaxes(np.swapaxes(a, 1, 4), 2, 4))

        assert_canonical_transpose("ZTXYC", a, np.swapaxes(np.moveaxis(a, 0, 2), 1, 4))
        assert_canonical_transpose("ZTCXY", a, np.swapaxes(np.moveaxis(a, 0, 2), 3, 4))
        assert_canonical_transpose("YXCZT", a, np.swapaxes(np.moveaxis(a, 0, 3), 0, 4))
        assert_canonical_transpose("XYCZT", a, np.swapaxes(np.moveaxis(a, 1, 3), 0, 4))
        assert_canonical_transpose("CZTXY", a, np.swapaxes(np.moveaxis(a, 2, 0), 3, 4))
        assert_canonical_transpose("CXYTZ", a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 4))
        assert_canonical_transpose("CXYZT", a, np.swapaxes(np.moveaxis(a, 4, 0), 2, 4))
        assert_canonical_transpose("CZXYT", a, np.swapaxes(np.moveaxis(a, 4, 0), 3, 4))
        assert_canonical_transpose("ZTYXC", a, np.swapaxes(np.moveaxis(a, 4, 1), 0, 2))
        assert_canonical_transpose("TXYZC", a, np.swapaxes(np.moveaxis(a, 4, 1), 2, 4))
        assert_canonical_transpose("CTYXZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 0, 1))
        assert_canonical_transpose("TXYCZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 1, 4))

        assert_canonical_transpose("XYTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 3))
        assert_canonical_transpose("YXTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 4))
        assert_canonical_transpose("TYXCZ", a, np.moveaxis(np.moveaxis(a, 1, 4), 1, 4))
        assert_canonical_transpose("ZYXTC", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 1))
        assert_canonical_transpose("CYXTZ", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 2))
        assert_canonical_transpose("TYXZC", a, np.moveaxis(np.moveaxis(a, 4, 1), 4, 2))
        assert_canonical_transpose("ZCYXT", a, np.moveaxis(np.moveaxis(a, 4, 0), 1, 2))
        assert_canonical_transpose("ZYXCT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 1))
        assert_canonical_transpose("CYXZT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 2))

    def test_transpose_squeeze(self):
        a = np.random.rand(1, 60, 40)
        assert_transpose("ZXY", "XY", a, np.squeeze(a, 0))
        assert_transpose("ZXY", "YX", a, np.swapaxes(np.squeeze(a, 0), 0, 1))

        a = np.random.rand(1, 1, 60, 40)
        assert_transpose("CZXY", "XY", a, np.squeeze(a, (0, 1)))
        assert_transpose("CZXY", "CXY", a, np.squeeze(a, 1))
        assert_transpose("CZXY", "ZXY", a, np.squeeze(a, 0))
        assert_transpose("CZXY", "YX", a, np.swapaxes(np.squeeze(a, (0, 1)), 0, 1))
        assert_transpose("CZXY", "CYX", a, np.swapaxes(np.squeeze(a, 1), 1, 2))
        assert_transpose("CZXY", "ZYX", a, np.swapaxes(np.squeeze(a, 0), 1, 2))

        a = np.random.rand(1, 1, 1, 60, 40)
        assert_transpose("TCZXY", "XY", a, np.squeeze(a, (0, 1, 2)))
        assert_transpose("TCZXY", "ZXY", a, np.squeeze(a, (0, 1)))
        assert_transpose("TCZXY", "CXY", a, np.squeeze(a, (0, 2)))
        assert_transpose("TCZXY", "TXY", a, np.squeeze(a, (1, 2)))
        assert_transpose("TCZXY", "CZXY", a, np.squeeze(a, 0))
        assert_transpose("TCZXY", "TZXY", a, np.squeeze(a, 1))
        assert_transpose("TCZXY", "TCXY", a, np.squeeze(a, 2))
        assert_transpose("TCZXY", "YX", a, np.swapaxes(np.squeeze(a, (0, 1, 2)), 0, 1))
        assert_transpose("TCZXY", "ZYX", a, np.swapaxes(np.squeeze(a, (0, 1)), 1, 2))
        assert_transpose("TCZXY", "CYX", a, np.swapaxes(np.squeeze(a, (0, 2)), 1, 2))
        assert_transpose("TCZXY", "TYX", a, np.swapaxes(np.squeeze(a, (1, 2)), 1, 2))
        assert_transpose("TCZXY", "CZYX", a, np.swapaxes(np.squeeze(a, 0), 2, 3))
        assert_transpose("TCZXY", "TZYX", a, np.swapaxes(np.squeeze(a, 1), 2, 3))
        assert_transpose("TCZXY", "TCYX", a, np.swapaxes(np.squeeze(a, 2), 2, 3))

    def test_transpose_expand(self):
        a = np.random.rand(10, 60, 40)
        assert_transpose("CYX", "ZCYX", a, np.expand_dims(a, 0))
        assert_transpose("CYX", "CZYX", a, np.expand_dims(a, 1))
        assert_transpose("CYX", "TZCYX", a, np.expand_dims(a, (0, 1)))
        assert_transpose("CYX", "TCZYX", a, np.expand_dims(a, (0, 2)))
        assert_transpose("CYX", "CTZYX", a, np.expand_dims(a, (1, 2)))
        assert_transpose("CYX", "ZCXY", a, np.swapaxes(np.expand_dims(a, 0), 2, 3))
        assert_transpose("CYX", "CZXY", a, np.swapaxes(np.expand_dims(a, 1), 2, 3))
        assert_transpose(
            "CYX", "TZCXY", a, np.swapaxes(np.expand_dims(a, (0, 1)), 3, 4)
        )
        assert_transpose(
            "CYX", "TCZXY", a, np.swapaxes(np.expand_dims(a, (0, 2)), 3, 4)
        )
        assert_transpose(
            "CYX", "CTZXY", a, np.swapaxes(np.expand_dims(a, (1, 2)), 3, 4)
        )


def assert_transpose(source, target, a, expected):
    transposed = transpose_array(a, source, target)
    np.testing.assert_array_equal(transposed, expected)


def assert_canonical_transpose(source, a, expected):
    if not isinstance(source, Axes):
        source = Axes(source)
    target = source.canonical(a.shape)
    assert_transpose(source.dims, target.dims, a, expected)
