"""
Implement the cluster analysis of CSM
=====================================


Conventions
-----------

fast changing axis is the x-axis
slowly changing axis is the y-axis
aka row-major ordering

"""
import numpy as np
cimport numpy as np


cdef struct Cluster:
    int ec
    int nc
    double xc
    double yc



cdef seeker(Cluster *cluster, np.ndarray image, int j, int i):
    pass


def cluster_analysis(int threshold, np.ndarray image, np.ndarray clusters):
    assert image.ndim == 2
    cdef iy_max = image.shape[0]
    cdef ix_max = image.shape[1]
    cdef int i
    cdef int j
    cdef Cluster cluster
    for j in range(iy_max):
        for i in range(ix_max):
            if image[j][i] > threshold:
                seeker(&cluster, image, j, i)

