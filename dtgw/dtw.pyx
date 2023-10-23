cimport cython

import numpy as np
cimport numpy as np


cdef inline double min3d(double a, double b, double c):
	if a <= b:
		return a if a <= c else c
	else:
		return b if b <= c else c


cdef inline int argmin3d(double a, double b, double c):
	if a <= b:
		return 0 if a <= c else 2
	else:
		return 1 if b <= c else 2


@cython.boundscheck(False)
@cython.wraparound(False)
def dynamic_timewarping(np.ndarray cost_matrix):
	"""Computes the dynamic timewarping distance between two sequences A and B
	where the distance between A[i] and B[j] is given by »cost_matrix[i,j]«.

	Returns:
	--------
	d : number
		The dynamic timewarping distance
	path : [np.array((x, y))]) 
		An optimal warping path as list of coordinate pairs
	"""
	cdef int n1, n2
	n1 = cost_matrix.shape[0]
	n2 = cost_matrix.shape[1]
	cdef np.ndarray[np.double_t, ndim=2] matrix
	matrix = np.full((n1+1, n2+1), np.inf)
	# the last row and column serve as guard rails
	matrix[:n1,:n2] = cost_matrix
	# Compute the warp distance of seq1[i:] and seq2[j:] and store
	# that value in matrix[i,j]:
	matrix[n1,n2] = 0
	cdef int i, j
	for i in range(n1-1, -1, -1):
		for j in range(n2-1, -1, -1):
			matrix[i,j] += min3d(
				matrix[i+1, j+1],
				matrix[i,   j+1],
				matrix[i+1, j  ]
			)
	# build warp path from (0, 0) to (n1-1, n2-1):
	cdef int x_step[3]
	x_step = (1, 1, 0)
	cdef int y_step[3]
	y_step = (1, 0, 1)
	cdef int direction
	i = j = 0
	cdef int l
	l = 1
	path = np.zeros((n1+n2, 2), dtype=int)
	# path = [(0, 0)]

	while i < n1 - 1 or j < n2 - 1:
		direction = argmin3d(matrix[i+1, j+1], matrix[i+1, j], matrix[i, j+1])
		i += x_step[direction]
		j += y_step[direction]
		# path.append((i, j))
		path[l,0] = i
		path[l,1] = j
		l += 1

	path = path[:l,:]
	return matrix[0,0], path
