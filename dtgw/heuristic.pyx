cimport cython

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def update_matchcost(metric, features1, features2, warppath):
	cdef int w, n
	w, n = warppath.shape[0], features1.shape[1]
	mc = np.zeros((n, n))
	for i in range(w):
		mc += metric(features1[warppath[i][0],:,:], features2[warppath[i][1],:,:])
	return mc


@cython.boundscheck(False)
@cython.wraparound(False)
def update_warpcost(metric, features1, features2, matching):
	cdef int t1, t2
	t1, t2 = features1.shape[0], features2.shape[0]
	wc = np.zeros((t1, t2))
	for v, w in enumerate(matching):
		wc += metric(features1[:,v,:], features2[:,w,:])
	return wc


# way slower than cdist?!
@cython.boundscheck(False)
@cython.wraparound(False)
def neighbor_metric(f1, f2):
	cdef int n, m
	n, m = f1.shape[0], f2.shape[0]
	res = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			res[i][j] += np.sum(np.abs(f1[i] - f2[j]))
	return res


@cython.boundscheck(False)
@cython.wraparound(False)
def label_matrices(tgraph):
	cdef int t, n
	t, n = tgraph.lifetime, tgraph.num_vertices()
	degrees = np.empty((t, n))
	red_neighbors = np.empty((t, n))
	vertices = tgraph.get_vertices()
	for i in range(t):
		l = tgraph.layer(i)
		for v in vertices:
			n_v = l.get_out_neighbors(v)
			# degrees[i, v] = n_v.shape[0]
			red_neighbors[i, v] = np.sum(tgraph.vertex_labels[i, n_v])
		degrees[i,:] = l.get_out_degrees(vertices)
	return degrees, red_neighbors


