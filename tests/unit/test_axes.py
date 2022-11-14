import itertools as it

import numpy as np
import pytest

from tiledb.bioimg.converters.axes import Axes, Move, Swap, minimize_transpositions


class TestTranspositions:
    @pytest.mark.parametrize(
        "s,i,j", [("ADCBE", 1, 3), ("DBCAE", 3, 0), ("ACBDE", 2, 1)]
    )
    def test_swap(self, s, i, j):
        swap = Swap(i, j)
        assert swap.transposed(s) == list("ABCDE")
        b = list(s)
        assert swap.transpose(b) is None
        assert b == list("ABCDE")

    @pytest.mark.parametrize(
        "s,i,j",
        [
            ("ADBCE", 1, 3),
            ("ACDBE", 3, 1),
            ("ACBDE", 1, 2),
            ("ACBDE", 2, 1),
            ("EABCD", 0, 4),
            ("BCDEA", 4, 0),
        ],
    )
    def test_move(self, s, i, j):
        move = Move(i, j)
        assert move.transposed(s) == list("ABCDE")
        b = list(s)
        assert move.transpose(b) is None
        assert b == list("ABCDE")

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
        s = list(s)
        for transposition in transpositions:
            transposition.transpose(s)
        assert s == list(t)


class TestAxes:
    def test_init(self):
        assert Axes("XYZ").members == "XYZ"

        with pytest.raises(ValueError) as excinfo:
            Axes("XYZW")
        assert str(excinfo.value) == "'W' is not a valid Axis"

        with pytest.raises(ValueError) as excinfo:
            Axes("XYZX")
        assert "Duplicate axes" in str(excinfo.value)

    @pytest.mark.parametrize(
        "canonical_axes", ["YX", "ZYX", "CYX", "TYX", "CZYX", "TCYX", "TZYX", "TCZYX"]
    )
    def test_canonical(self, canonical_axes):
        for axes in map(Axes, it.permutations(canonical_axes)):
            assert axes.canonical().members == canonical_axes

    def test_transpose_2d(self):
        a = np.random.rand(60, 40)
        assert_transpose("YX", a, a)
        assert_transpose("XY", a, np.swapaxes(a, 0, 1))

    def test_transpose_3d(self):
        a = np.random.rand(10, 60, 40)
        for s in "ZYX", "CYX", "TYX":
            assert_transpose(s, a, a)
        for s in "XYZ", "XYC", "XYT":
            assert_transpose(s, a, np.swapaxes(a, 0, 2))
        for s in "ZXY", "CXY", "TXY":
            assert_transpose(s, a, np.swapaxes(a, 1, 2))
        for s in "YXZ", "YXC", "YXT":
            assert_transpose(s, a, np.moveaxis(a, 2, 0))

    def test_transpose_4d(self):
        a = np.random.rand(3, 10, 60, 40)
        for s in "CZYX", "TCYX", "TZYX":
            assert_transpose(s, a, a)
        for s in "ZCYX", "CTYX", "ZTYX":
            assert_transpose(s, a, np.swapaxes(a, 0, 1))
        for s in "CZXY", "TCXY", "TZXY":
            assert_transpose(s, a, np.swapaxes(a, 2, 3))
        for s in "ZYXC", "CYXT", "ZYXT":
            assert_transpose(s, a, np.moveaxis(a, 3, 0))
        for s in "CYXZ", "TYXC", "TYXZ":
            assert_transpose(s, a, np.moveaxis(a, 3, 1))
        for s in "YXZC", "YXCT", "YXZT":
            assert_transpose(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 0))
        for s in "YXCZ", "YXTC", "YXTZ":
            assert_transpose(s, a, np.moveaxis(np.moveaxis(a, 2, 0), 3, 1))
        for s in "ZCXY", "CTXY", "ZTXY":
            assert_transpose(s, a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 3))
        for s in "XYZC", "XYCT", "XYZT":
            assert_transpose(s, a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 2))
        for s in "CXYZ", "TXYC", "TXYZ":
            assert_transpose(s, a, np.swapaxes(np.moveaxis(a, 3, 1), 2, 3))
        for s in "ZXYC", "CXYT", "ZXYT":
            assert_transpose(s, a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 3))
        for s in "XYCZ", "XYTC", "XYTZ":
            assert_transpose(s, a, np.swapaxes(np.moveaxis(a, 2, 0), 1, 3))

    def test_transpose_5d(self):
        a = np.random.rand(7, 3, 10, 60, 40)
        assert_transpose("TCZYX", a, a)

        assert_transpose("CTZYX", a, np.swapaxes(a, 0, 1))
        assert_transpose("ZCTYX", a, np.swapaxes(a, 0, 2))
        assert_transpose("TZCYX", a, np.swapaxes(a, 1, 2))
        assert_transpose("TCXYZ", a, np.swapaxes(a, 2, 4))
        assert_transpose("TCZXY", a, np.swapaxes(a, 3, 4))

        assert_transpose("ZTCYX", a, np.moveaxis(a, 0, 2))
        assert_transpose("CZTYX", a, np.moveaxis(a, 2, 0))
        assert_transpose("CZYXT", a, np.moveaxis(a, 4, 0))
        assert_transpose("TZYXC", a, np.moveaxis(a, 4, 1))
        assert_transpose("TCYXZ", a, np.moveaxis(a, 4, 2))

        assert_transpose("CTXYZ", a, np.swapaxes(np.swapaxes(a, 0, 1), 2, 4))
        assert_transpose("CTZXY", a, np.swapaxes(np.swapaxes(a, 0, 1), 3, 4))
        assert_transpose("ZCTXY", a, np.swapaxes(np.swapaxes(a, 0, 2), 3, 4))
        assert_transpose("YXZTC", a, np.swapaxes(np.swapaxes(a, 0, 3), 1, 4))
        assert_transpose("XYZCT", a, np.swapaxes(np.swapaxes(a, 0, 4), 1, 3))
        assert_transpose("ZCXYT", a, np.swapaxes(np.swapaxes(a, 0, 4), 2, 4))
        assert_transpose("TZCXY", a, np.swapaxes(np.swapaxes(a, 1, 2), 3, 4))
        assert_transpose("TZXYC", a, np.swapaxes(np.swapaxes(a, 1, 4), 2, 4))

        assert_transpose("ZTXYC", a, np.swapaxes(np.moveaxis(a, 0, 2), 1, 4))
        assert_transpose("ZTCXY", a, np.swapaxes(np.moveaxis(a, 0, 2), 3, 4))
        assert_transpose("YXCZT", a, np.swapaxes(np.moveaxis(a, 0, 3), 0, 4))
        assert_transpose("XYCZT", a, np.swapaxes(np.moveaxis(a, 1, 3), 0, 4))
        assert_transpose("CZTXY", a, np.swapaxes(np.moveaxis(a, 2, 0), 3, 4))
        assert_transpose("CXYTZ", a, np.swapaxes(np.moveaxis(a, 3, 0), 2, 4))
        assert_transpose("CXYZT", a, np.swapaxes(np.moveaxis(a, 4, 0), 2, 4))
        assert_transpose("CZXYT", a, np.swapaxes(np.moveaxis(a, 4, 0), 3, 4))
        assert_transpose("ZTYXC", a, np.swapaxes(np.moveaxis(a, 4, 1), 0, 2))
        assert_transpose("TXYZC", a, np.swapaxes(np.moveaxis(a, 4, 1), 2, 4))
        assert_transpose("CTYXZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 0, 1))
        assert_transpose("TXYCZ", a, np.swapaxes(np.moveaxis(a, 4, 2), 1, 4))

        assert_transpose("XYTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 3))
        assert_transpose("YXTCZ", a, np.moveaxis(np.moveaxis(a, 0, 4), 0, 4))
        assert_transpose("TYXCZ", a, np.moveaxis(np.moveaxis(a, 1, 4), 1, 4))
        assert_transpose("ZYXTC", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 1))
        assert_transpose("CYXTZ", a, np.moveaxis(np.moveaxis(a, 3, 0), 4, 2))
        assert_transpose("TYXZC", a, np.moveaxis(np.moveaxis(a, 4, 1), 4, 2))
        assert_transpose("ZCYXT", a, np.moveaxis(np.moveaxis(a, 4, 0), 1, 2))
        assert_transpose("ZYXCT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 1))
        assert_transpose("CYXZT", a, np.moveaxis(np.moveaxis(a, 4, 0), 4, 2))


def assert_transpose(s, a, b):
    np.testing.assert_array_equal(Axes(s).transpose(a), b)
