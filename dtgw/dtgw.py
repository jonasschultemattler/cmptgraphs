#!/usr/bin/env python3

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double

array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

libcd = npct.load_library("dtgw.so", ".")


libcd.dtgw.restype = c_double
libcd.dtgw.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

def dtgw_wrapper(f1, f2, init, window, max_iterations):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	if init == "diagonal_warping":
		i = 1
	elif init == "optimistic_warping":
		i = 2
	elif init == "optimistic_matching":
		i = 3
	elif init == "sigma*":
		i = 4
	else:
		raise ValueError("Invalid initialization")
	return libcd.dtgw(f1, f2, t1, t2, n, c, i, window, max_iterations)


def compute_dtgw(features1, features2, eps, init="diagonal_warping", window=None, max_iterations=1000):
	t1, n1, k1 = features1.shape
	t2, n2, k2 = features2.shape
	n = max(n1, n2)
	features1 = np.hstack((features1, np.broadcast_to(eps, (t1, n-n1, k1))))
	features2 = np.hstack((features2, np.broadcast_to(eps, (t2, n-n2, k2))))

	cost = dtgw_wrapper(features1, features2, init, window, max_iterations)
	
	return cost


