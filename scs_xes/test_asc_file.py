import unittest
from io import BytesIO
import numpy as np
from scs_xes.asc_file import AscFile

TEST_ASC = b'''\
1\t1\t2\t-3\r
2\t3\t4\t5\r
1\t-6\t7\t8\r
2\t-9\t10\t11\r
'''


class TestAscFile(unittest.TestCase):
    def setUp(self):
        self.obj = AscFile(BytesIO(TEST_ASC), num_rows=2)

    def test1(self):
        it = iter(self.obj)
        i1 = next(it)
        i2 = next(it)
        self.assertRaises(StopIteration, next, it)

    def test2(self):
        it = iter(self.obj)
        i1 = next(it)
        np.testing.assert_array_equal(i1.shape, (2, 3))
        np.testing.assert_array_equal(i1, [[1, 2, -3], [3, 4, 5]])

    def test3(self):
        it = iter(self.obj)
        i1 = next(it)
        i2 = next(it)
        np.testing.assert_array_equal(i2.shape, (2, 3))
        np.testing.assert_array_equal(i2, [[-6, 7, 8], [-9, 10, 11]])

    def test4(self):
        it = iter(self.obj)
        i1a = next(it)
        it = iter(self.obj)
        i1b = next(it)
        np.testing.assert_array_equal(i1a, i1b)