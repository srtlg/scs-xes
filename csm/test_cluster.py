import unittest
import numpy as np
from io import StringIO
from cluster import cluster_analysis


TEST_IMAGE_1 = """\
 . . . . . . # 0
 . . 3 . . . # 1
 . . 5 2 . . # 2
 . . . 1 . . # 3
#0 1 2 3 4 5
"""


class TestClusterSingle(unittest.TestCase):
    def setUp(self):
        self.image = np.loadtxt(StringIO(TEST_IMAGE_1.replace('.', '0')), dtype=np.int16)

    def test(self):
        clusters = cluster_analysis(self.image)
        self.assertEqual(1, len(clusters))
        cluster = clusters[0]
        self.assertEqual(3+5+2+1, cluster.ec)
        self.assertAlmostEqual((3.0*2 + 5*2 + 2*3 + 1*3)/cluster.ec, cluster.xc)
        self.assertAlmostEqual((3.0*1 + 5*2 + 2*2 + 1*3)/cluster.ec, cluster.yc)
        self.assertEqual(4, cluster.nc)


# adopted from Figure 3.16 of [Trassinelli2005]_
TEST_IMAGE_2 = """\
 . . . . . . 1 2 . . . 4 . . . . . . . . . . . . # 0
 . . . . . . . . . 5 6 3 . . . . . . . . . . . . # 1
 . . 5 . . . . . 7 5 . . . . . . . . . . . . . . # 2
 . . . . . . 3 4 4 . . . . . . 3 . . . . 3 . . . # 3
 . . . . . 6 6 . . . . . . . . . . . . . 2 . . . # 4
 . . . 3 6 . . . . 5 . . . . . . . . . . . . . . # 5
 . 3 4 5 . . . 2 . 4 3 . . . . . . . . . . . . . # 6
 3 2 . . . . . 1 . . . . . . . . . . . . . . . . # 7
 5 . . . . . . . . . . . . . . . 7 . . . . 1 7 2 # 8
 . . 2 . . . . . . . 5 . . . . . 7 6 . . . . 3 . # 9
 . . 2 . . . . . . . . . . . . . . 3 6 . . . . . #10
 . . . . . . . . 5 . . . . . . . . . 1 . . . . . # 1
 . . . . . . . . . . . . . . . . . . . . . . . . # 2
#0                   1                   2
#0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
"""


class TestClusterFig316(unittest.TestCase):
    def setUp(self):
        self.image = np.loadtxt(StringIO(TEST_IMAGE_2.replace('.', '0')), dtype=np.int16)

    def test(self):
        clusters = cluster_analysis(self.image)
        self.assertEqual(13, len(clusters))


TEST_IMAGE_3 = """\
 . . . . . # 0
 . . . . . # 1
 . . 1 7 2 # 2
 . . . 3 . # 3
 . . . . . # 4
 . . . . . # 5
#0 1 2 3 4
"""

class TestClusterTee(unittest.TestCase):
    def setUp(self):
        self.image = np.loadtxt(StringIO(TEST_IMAGE_3.replace('.', '0')), dtype=np.int16)

    def test(self):
        clusters = cluster_analysis(self.image)
        self.assertEqual(1, len(clusters))
        cluster = clusters[0]
        self.assertEqual(1+7+2+3, cluster.ec)
        self.assertAlmostEqual((1.0*2 + 7*3 + 3*3 + 2*4)/cluster.ec, cluster.xc)
        self.assertAlmostEqual((1.0*2 + 7*2 + 2*2 + 3*3)/cluster.ec, cluster.yc)
        self.assertEqual(4, cluster.nc)


TEST_IMAGE_4 = """\
 . . . 3 6 . . # 0
 . 3 4 5 . . . # 1
 3 2 . . . . . # 2
 5 . . . . . . # 3
#0 1 2 3 4 5 6 
"""

class TestClusterLineSouthWest(unittest.TestCase):
    def setUp(self):
        self.image = np.loadtxt(StringIO(TEST_IMAGE_4.replace('.', '0')), dtype=np.int16)

    def test(self):
        clusters = cluster_analysis(self.image)
        self.assertEqual(1, len(clusters))
        cluster = clusters[0]
        self.assertEqual(np.sum(self.image), cluster.ec)
        self.assertAlmostEqual(float((3+5)*0 + (3+2)*1 + 4*2 + (3+5)*3 + 6*4)/cluster.ec, cluster.xc)
        self.assertAlmostEqual(float((3+6)*0 + (3+4+5)*1 + (3+2)*2 + 5*3)/cluster.ec, cluster.yc)
        self.assertEqual(8, cluster.nc)
