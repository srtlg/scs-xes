"""
Implement the cluster analysis of CSM
=====================================


Conventions
-----------

fast changing axis is the x-axis
slowly changing axis is the y-axis
aka row-major ordering

References
----------

.. [Trassinelli2005] M. Trassinelli, "Quantum Electrodynamics Tests and X-rays Standards using Pionic Atoms and
Highly Charged Ions" Université Pierre et Marie Curie, 2005.  https://tel.archives-ouvertes.fr/tel-00067768v2
"""
import numpy as np
cimport cython
cimport numpy as np
from libc.string cimport memset


cdef struct Cluster:
    int ec
    int nc
    double xc
    double yc

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef void seeker(Cluster *cluster, np.ndarray image, int j, int i, int threshold):
    cdef int ec0 = cluster.ec
    #print(cluster.nc, cluster.ec, cluster.xc, cluster.yc, ':', j, i, threshold, '|', image[j, i])
    cluster.nc += cluster.nc
    cluster.ec += image[j, i]
    cluster.xc = (cluster.xc * ec0 + i * image[j, i]) / cluster.ec
    cluster.yc = (cluster.yc * ec0 + j * image[j, i]) / cluster.ec
    image[j, i] = 0 # fixme assumes that threshold >= 0
    if (i - 1) >= 0 and image[j, i - 1] > threshold:
        seeker(cluster, image, j, i - 1, threshold)
    elif (i + 1) < image.shape[1] and image[j, i + 1] > threshold:
        seeker(cluster, image, j, i + 1, threshold)
    elif (j - 1) >= 0 and image[j - 1, i] > threshold:
        seeker(cluster, image, j - 1, i, threshold)
    elif (j + 1) < image.shape[0] and image[j + 1, i] > threshold:
        seeker(cluster, image, j + 1, i, threshold)


def cluster_analysis(np.ndarray image, int threshold=0):
    """
    Run cluster analysis on 2d-detector output
    :param image: the image to be analysed
    :param threshold: the threshold for colouring the pixels
    :return: the found clusters
    """
    assert image.ndim == 2
    cdef iy_max = image.shape[0]
    cdef ix_max = image.shape[1]
    cdef int i
    cdef int j
    cdef int ic = 0
    cdef Cluster cluster
    cdef image_work = image.copy()
    for j in range(iy_max):
        for i in range(ix_max):
            if image[j][i] > threshold:
                memset(&cluster, 0, sizeof(Cluster))
                seeker(&cluster, image_work, j, i, threshold)
                ic += 1
                return cluster
