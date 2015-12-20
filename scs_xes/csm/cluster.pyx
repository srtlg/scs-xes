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
from __future__ import print_function
import numpy as np
cimport cython
cimport numpy as np
from libc.string cimport memset
from libc.stdint cimport int32_t, int16_t


Cluster_dtype = np.dtype([
    ('ec', np.int32),
    ('image', np.int16),
    ('nc', np.int16),
    ('xc', np.float32),
    ('yc', np.float32),
    ])


cdef enum:
    BLOCK_SIZE = 1024
    NOT_CHARGED = 0


cdef packed struct Cluster_t:
    int32_t ec
    int16_t image_index
    int16_t nc
    float xc
    float yc


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef void seeker(Cluster_t *cluster, np.ndarray image, int j, int i, int threshold):
    cdef int ec0 = cluster.ec
    cluster.nc += 1
    cluster.ec += image[j, i]
    cluster.xc = (cluster.xc * ec0 + i * image[j, i]) / cluster.ec
    cluster.yc = (cluster.yc * ec0 + j * image[j, i]) / cluster.ec
    #print(cluster.nc, cluster.ec, cluster.xc, cluster.yc, ':', j, i, threshold, '|', image[j, i])
    image[j, i] = NOT_CHARGED
    if (i - 1) >= 0 and image[j, i - 1] > threshold:
        seeker(cluster, image, j, i - 1, threshold)
    if (i + 1) < image.shape[1] and image[j, i + 1] > threshold:
        seeker(cluster, image, j, i + 1, threshold)
    if (j - 1) >= 0 and image[j - 1, i] > threshold:
        seeker(cluster, image, j - 1, i, threshold)
    if (j + 1) < image.shape[0] and image[j + 1, i] > threshold:
        seeker(cluster, image, j + 1, i, threshold)


def cluster_analysis(np.ndarray image, int16_t image_index=-1, int threshold=0):
    """
    Run cluster analysis on 2d-detector output
    :param image: [in] the image to be analysed
    :param image_index: [in] the image index (for identifying the photon event later)
    :param threshold: [in] the threshold for colouring the pixels
    :return: the found clusters
    """
    assert threshold >= 0, 'threshold too low, adjust NOT_CHARGED'
    assert image.ndim == 2, 'this routine expects a two dimensional image, got %d' % image.ndim
    cdef int iy_max = image.shape[0]
    cdef int ix_max = image.shape[1]
    cdef int i
    cdef int j
    cdef int ic = 0
    cdef np.ndarray image_work = image.copy()
    cdef np.ndarray[Cluster_t, ndim=1, mode="c"] cluster_array
    cdef Cluster_t[:] cluster_view
    cluster_array = np.recarray(shape=(BLOCK_SIZE,), dtype=Cluster_dtype)
    cluster_view = cluster_array
    memset(&cluster_array[0], 0, cluster_array.size * cluster_array.itemsize)
    for j in range(iy_max):
        for i in range(ix_max):
            if image_work[j, i] > threshold:
                #print('>>>>', ic, '[%4d,%4d,] = %5d' % (j, i, image_work[j, i]), '|', len(cluster_array))
                cluster_view[ic].image_index = image_index
                seeker(&cluster_view[ic], image_work, j, i, threshold)
                ic += 1
                if ic >= cluster_array.size:
                    cluster_array.resize((ic + BLOCK_SIZE,), refcheck=False)
                    cluster_view = cluster_array
                    # see documentation of numpy.resize: ndarray.resize fills
                    # with zeros
    cluster_array.resize((ic,), refcheck=False)
    return cluster_array

