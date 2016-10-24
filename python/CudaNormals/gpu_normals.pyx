# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "cuda/compute_normals.h":
    void CudaNormals::compute_normals(np.float32_t*, np.float32_t*, int, int, int)

def gpu_normals(np.ndarray[np.float32_t, ndim=2] points, np.int32_t w, np.int32_t h, np.int32_t device_id=0):
    cdef int points_dim = points.shape[0]
    cdef int points_num = points.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] \
        normals = np.zeros((points_num, points_dim), dtype=np.float32)

    _compute_normals(&normals[0, 0], &points[0, 0], w, h, device_id)
    return normals
