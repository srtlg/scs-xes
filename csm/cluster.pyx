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


Cluster_dtype = np.dtype([
    ('ec', np.int32),
    ('nc', np.int32),
    ('xc', np.float64),
    ('yc', np.float64),
    ])


cdef enum:
    BLOCK_SIZE = 2


cdef packed struct Cluster_t:
    int ec
    int nc
    double xc
    double yc


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef void seeker(Cluster_t *cluster, np.ndarray image, int j, int i, int threshold):
    cdef int ec0 = cluster.ec
    #print(cluster.nc, cluster.ec, cluster.xc, cluster.yc, ':', j, i, threshold, '|', image[j, i])
    cluster.nc += 1
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
    :param image: [in] the image to be analysed
    :param threshold: [in] the threshold for colouring the pixels
    :return: the found clusters
    """
    assert image.ndim == 2
    cdef iy_max = image.shape[0]
    cdef ix_max = image.shape[1]
    cdef int i
    cdef int j
    cdef int ic = 0
    cdef image_work = image.copy()
    cdef np.ndarray[Cluster_t, ndim=1] cluster_array
    cluster_array = np.recarray(shape=(BLOCK_SIZE,), dtype=Cluster_dtype)
    memset(&cluster_array[0], 0, cluster_array.size * cluster_array.itemsize)
    for j in range(iy_max):
        for i in range(ix_max):
            if image_work[j][i] > threshold:
                seeker(&cluster_array[ic], image_work, j, i, threshold)
                ic += 1
                if ic >= cluster_array.size:
                    cluster_array.resize((ic + BLOCK_SIZE,), refcheck=False)
                    memset(&cluster_array[ic], 0, BLOCK_SIZE * cluster_array.itemsize)
    cluster_array.resize((ic,), refcheck=False)
    return cluster_array

