#!/usr/bin/env python3

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double, c_float


def init_to_number(init):
	if init == "diagonal_warping":
		return 1
	elif init == "optimistic_warping":
		return 2
	elif init == "optimistic_matching":
		return 3
	elif init == "sigma*":
		return 4
	else:
		raise ValueError("Invalid initialization")


def metric_to_number(metric):
	if metric == "l1":
		return 1
	elif metric == "l2":
		return 2
	elif metric == "dot":
		return 3
	else:
		raise ValueError("Invalid initialization")


array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')
array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3, flags='CONTIGUOUS')
array_2d_bool = npct.ndpointer(dtype=bool, ndim=2, flags='CONTIGUOUS')
array_3d_bool = npct.ndpointer(dtype=bool, ndim=3, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library("dtgw.so", "dtgw/build")


libcd.vertex_feature_subtrees0.restype = None
libcd.vertex_feature_subtrees0.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_subtrees1.restype = None
libcd.vertex_feature_subtrees1.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_subtrees2.restype = None
libcd.vertex_feature_subtrees2.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]


def subtrees_feature_wrapper(tadj, tlabels, k):
	t, n = tlabels.shape
	if k == 0:
		feature = np.zeros((t, n, 1)).astype('float32')
		libcd.vertex_feature_subtrees0(tadj, tlabels, n, t, feature)
	elif k == 1:
		feature = np.zeros((t, n, 3)).astype('float32')
		libcd.vertex_feature_subtrees1(tadj, tlabels, n, t, feature)
	elif k == 2:
		feature = np.zeros((t, n, 5)).astype('float32')
		libcd.vertex_feature_subtrees2(tadj, tlabels, n, t, feature)
	else:
		return None
	return feature.astype('float32')


libcd.vertex_feature_walks1.restype = None
libcd.vertex_feature_walks1.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_walks2.restype = None
libcd.vertex_feature_walks2.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_walks3.restype = None
libcd.vertex_feature_walks3.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]


def walks_feature_wrapper(tadj, tlabels, k):
	t, n = tlabels.shape
	if k == 1:
		feature = np.zeros((t, n, 4)).astype('float32')
		libcd.vertex_feature_walks1(tadj, tlabels, n, t, feature)
	elif k == 2:
		# feature = np.zeros((t, n, 8)).astype('float32')
		# libcd.vertex_feature_walks2(tadj, tlabels, n, t, feature)
		feature = np.zeros((t, n, 12)).astype('float32')
		libcd.vertex_feature_walks2(tadj, tlabels, n, t, feature)
	# elif k == 3:
	# 	feature = np.zeros((t, n, 16)).astype('float32')
	# 	libcd.vertex_feature_walks3(tadj, tlabels, n, t, feature)
	else:
		return None
	return feature.astype('float32')



libcd.vertex_feature_neighbors1.restype = None
libcd.vertex_feature_neighbors1.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_neighbors2.restype = None
libcd.vertex_feature_neighbors2.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]


def neighbors_feature_wrapper(tadj, tlabels, k):
	t, n = tlabels.shape
	if k == 1:
		feature = np.zeros((t, n, 3)).astype('float32')
		libcd.vertex_feature_neighbors1(tadj, tlabels, n, t, feature)
	elif k == 2:
		feature = np.zeros((t, n, 5)).astype('float32')
		libcd.vertex_feature_neighbors2(tadj, tlabels, n, t, feature)
	else:
		return None
	return feature.astype('float32')




libcd.vertex_feature2_subtrees1.restype = None
libcd.vertex_feature2_subtrees1.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature2_subtrees2.restype = None
libcd.vertex_feature2_subtrees2.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]


def subtrees_feature_wrapper2(tadj, tlabels, k):
	t, n = tlabels.shape
	if k == 0:
		feature = np.zeros((t, n, 1)).astype('float32')
		libcd.vertex_feature_subtrees0(tadj, tlabels, n, t, feature)
	elif k == 1:
		feature = np.zeros((t, n, 3)).astype('float32')
		libcd.vertex_feature2_subtrees1(tadj, tlabels, n, t, feature)
	elif k == 2:
		feature = np.zeros((t, n, 5)).astype('float32')
		libcd.vertex_feature2_subtrees2(tadj, tlabels, n, t, feature)
	else:
		return None
	return feature.astype('float32')




libcd.vertex_feature_interaction2.restype = None
libcd.vertex_feature_interaction2.argtypes = [array_3d_bool, c_int, c_int, array_3d_float]
libcd.vertex_feature_interaction3.restype = None
libcd.vertex_feature_interaction3.argtypes = [array_3d_bool, c_int, c_int, array_3d_float]


def interaction_feature_wrapper(tadj, k):
	t, n, _ = tadj.shape
	if k == 1:
		feature = tadj
	elif k == 2:
		feature = np.zeros((t, n, 2*n)).astype('float32')
		libcd.vertex_feature_interaction2(tadj, n, t, feature)
	elif k == 3:
		feature = np.zeros((t, n, 3*n)).astype('float32')
		libcd.vertex_feature_interaction3(tadj, n, t, feature)
	else:
		return None
	return feature.astype('float32')


libcd.vertex_feature_degree2.restype = None
libcd.vertex_feature_degree2.argtypes = [array_3d_bool, array_2d_int, c_int, c_int, array_3d_float]
libcd.vertex_feature_degree3.restype = None
libcd.vertex_feature_degree3.argtypes = [array_3d_bool, array_2d_int, c_int, c_int, array_3d_float]
libcd.vertex_feature_degree4.restype = None
libcd.vertex_feature_degree4.argtypes = [array_3d_bool, array_2d_int, c_int, c_int, array_3d_float]


def degree_feature_wrapper(tadj, k):
	t, n, _ = tadj.shape
	if k == 1:
		feature = np.sum(tadj, axis=2)[:,:,np.newaxis]
	elif k == 2:
		feature = np.zeros((t, n, n+1)).astype('float32')
		degs = np.sum(tadj, axis=2).astype('int32')
		libcd.vertex_feature_degree2(tadj, degs, n, t, feature)
	elif k == 3:
		feature = np.zeros((t, n, 2*n+1)).astype('float32')
		degs = np.sum(tadj, axis=2).astype('int32')
		libcd.vertex_feature_degree3(tadj, degs, n, t, feature)
	elif k == 4:
		feature = np.zeros((t, n, 3*n+1)).astype('float32')
		degs = np.sum(tadj, axis=2).astype('int32')
		libcd.vertex_feature_degree4(tadj, degs, n, t, feature)
	else:
		return None
	return feature.astype('float32')




# libcd.vertex_feature_neighbors.restype = None
# libcd.vertex_feature_neighbors.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]

# def vertex_feature_neighbors_wrapper(tadj, tlabels):
# 	t, n = tlabels.shape
# 	feature = np.zeros((t, n, 3)).astype('float32')
# 	libcd.vertex_feature_neighbors(tadj, tlabels, n, t, feature)
# 	return feature.astype('float32')


# libcd.vertex_feature_neighbors2.restype = None
# libcd.vertex_feature_neighbors2.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_float]

# def vertex_feature_neighbors2_wrapper(tadj, tlabels):
# 	t, n = tlabels.shape
# 	feature = np.zeros((t, n, 5)).astype('float32')
# 	libcd.vertex_feature_neighbors2(tadj, tlabels, n, t, feature)
# 	return feature.astype('float32')





libcd.init_diagonal_warping2.restype = c_int
libcd.init_diagonal_warping2.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, array_2d_int, array_2d_float, c_int]

def init_diagonal_warping_wrapper(f1, f2, metric):
	t1, t2 = f1.shape[0], f2.shape[0]
	n, c = f1.shape[1], f1.shape[2]
	m = metric_to_number(metric)
	path = np.zeros((t1+t2, 2)).astype('int32')
	res = np.zeros((n, n)).astype('float32')
	l = libcd.init_diagonal_warping2(f1, f2, t1, t2, n, c, path, res, m)
	return res, path[:l,:]


libcd.init_diagonal_warping3.restype = c_int
libcd.init_diagonal_warping3.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, c_int, array_2d_int, array_2d_float, c_int]
libcd.init_diagonal_warping4.restype = c_int
libcd.init_diagonal_warping4.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, c_int, c_int, array_2d_int, array_2d_float, c_int]
libcd.init_diagonal_warping5.restype = c_int
libcd.init_diagonal_warping5.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, c_int, c_int, array_2d_int, array_2d_float, c_int]


def init_diagonal_warping_wrapper2(f1, f2, metric, dcost):
	t1, t2 = f1.shape[0], f2.shape[0]
	n1, n2, c = f1.shape[1], f2.shape[1], f1.shape[2]
	m = metric_to_number(metric)
	path = np.zeros((t1+t2, 2)).astype('int32')
	res = np.zeros((n1, n1)).astype('float32')
	if dcost == 0:
		l = libcd.init_diagonal_warping3(f1, f2, t1, t2, n1, n2, c, path, res, m)
	elif dcost == 1:
		z = np.zeros((n1, c)).astype('float32')
		l = libcd.init_diagonal_warping4(f1, f2, z, t1, t2, n1, n2, c, path, res, m)
	if dcost == 2:
		z = np.zeros((n1, c)).astype('float32')
		l = libcd.init_diagonal_warping5(f1, f2, z, t1, t2, n1, n2, c, path, res, m)
	# print(res)
	return res, path[:l,:]


libcd.sakoe_chiba_band.restype = None
libcd.sakoe_chiba_band.argtypes = [c_int, c_int, c_int, array_2d_int]

def sakoe_chiba_band_wrapper(t1, t2, window):
	region = np.zeros((2, t1), dtype=int).astype('int32')
	libcd.sakoe_chiba_band(t1, t2, window, region)
	return region



libcd.tgw.restype = c_double
libcd.tgw.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, c_int, c_int]

def tgw_wrapper(f1, f2, metric, window):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	i = metric_to_number(metric)
	return libcd.tgw(f1.astype('float32'), f2.astype('float32'), t1, t2, n, c, window, i)


# libcd.dtgw.restype = c_double
# libcd.dtgw.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

# def dtgw_wrapper(f1, f2, init, window, max_iterations):
# 	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
# 	i = init_to_number(init)
# 	return libcd.dtgw(f1, f2, t1, t2, n, c, i, window, max_iterations)
libcd.dtgw.restype = c_double
libcd.dtgw.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

def dtgw_wrapper(f1, f2, init, metric, window, max_iterations):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	i = init_to_number(init)
	m = metric_to_number(metric)
	return libcd.dtgw(f1.astype('float32'), f2.astype('float32'), m, t1, t2, n, c, i, window, max_iterations)

libcd.dtgw_log.restype = c_double
libcd.dtgw_log.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, array_1d_int]

def dtgw_log_wrapper(f1, f2, init, metric, window, max_iterations):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	i = init_to_number(init)
	m = metric_to_number(metric)
	res = np.zeros(2).astype('int32')
	d = libcd.dtgw_log(f1.astype('float32'), f2.astype('float32'), m, t1, t2, n, c, i, window, max_iterations, res)
	return d, res[0], res[1]


def compute_dtgw(features1, features2, eps, init="diagonal_warping", metric="l1", window=None, max_iterations=1000, log=False):
	t1, n1, k1 = features1.shape
	t2, n2, k2 = features2.shape
	n = max(n1, n2)
	features1 = np.hstack((features1, np.broadcast_to(eps, (t1, n-n1, k1)))).astype('float32')
	features2 = np.hstack((features2, np.broadcast_to(eps, (t2, n-n2, k2)))).astype('float32')

	if log:
		return dtgw_log_wrapper(features1, features2, init, metric, window, max_iterations)
	else:
		return dtgw_wrapper(features1, features2, init, metic, window, max_iterations)


libcd.dynamic_timewarping.restype = c_int
libcd.dynamic_timewarping.argtypes = [array_2d_float, c_int, c_int, array_2d_int, array_2d_int, array_2d_float]

def dtw_wrapper(cost_matrix, region):
	t1, t2 = cost_matrix.shape
	res = np.full((t1+1, t2+1), np.inf).astype('float32')
	region = region.astype('int32')
	path = np.zeros((t1*t2, 2), dtype=int).astype('int32')
	l = libcd.dynamic_timewarping(cost_matrix.astype('float32'), t1, t2, region, path, res)
	path = path[:l,:]
	return res[0,0], path


libcd.update_warping2.restype = None
libcd.update_warping2.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, array_1d_int, array_2d_int, array_2d_float, c_int]

def warping_update_wrapper(f1, f2, matching, region, metric):
	m = metric_to_number(metric)
	t1, t2, c = f1.shape[0], f2.shape[0], f1.shape[2]
	res = np.full((t1, t2), np.inf).astype('float32')
	f1 = f1.astype('float32')
	f2 = f2.astype('float32')
	region = region.astype('int32')
	matching = matching.astype('int32')
	libcd.update_warping2(f1, f2, t1, t2, matching.shape[0], c, matching, region, res, m)
	return res


libcd.update_warping3.restype = None
libcd.update_warping3.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, c_int, c_int, array_1d_int, array_2d_int, array_2d_float, c_int]
libcd.update_warping4.restype = None
libcd.update_warping4.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, c_int, c_int, array_1d_int, array_2d_int, array_2d_float, c_int]
libcd.update_warping5.restype = None
libcd.update_warping5.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, c_int, c_int, array_1d_int, array_2d_int, array_2d_float, c_int]


def warping_update_wrapper2(f1, f2, matching, region, metric, dcost):
	m = metric_to_number(metric)
	t1, t2, c = f1.shape[0], f2.shape[0], f1.shape[2]
	n1, n2 = f1.shape[1], f2.shape[1]
	res = np.full((t1, t2), np.inf).astype('float32')
	f1 = f1.astype('float32')
	f2 = f2.astype('float32')
	region = region.astype('int32')
	matching = matching.astype('int32')
	if dcost == 0:
		libcd.update_warping3(f1, f2, t1, t2, n1, n2, c, matching, region, res, m)
	elif dcost == 1:
		z = np.zeros((n1, c)).astype('float32')
		libcd.update_warping4(f1, f2, z, t1, t2, n1, n2, c, matching, region, res, m)
	elif dcost == 2:
		z = np.zeros((n1, c)).astype('float32')
		libcd.update_warping5(f1, f2, z, t1, t2, n1, n2, c, matching, region, res, m)
	# print(res)
	return res



libcd.update_matchcost2.restype = None
# libcd.update_matchcost2.argtypes = [array_3d_double, array_3d_double, c_int, c_int, array_2d_int, c_int, array_2d_double, c_int]
libcd.update_matchcost2.argtypes = [array_3d_float, array_3d_float, c_int, c_int, array_2d_int, c_int, array_2d_float, c_int]

def update_matchcost_wrapper(f1, f2, warppath, metric):
	m = metric_to_number(metric)
	n, c = f1.shape[1], f1.shape[2]
	warppath = warppath.astype('int32')
	res = np.zeros((n, n)).astype('float32')
	libcd.update_matchcost2(f1, f2, n, c, warppath, warppath.shape[0], res, m)
	return res


libcd.update_matchcost3.restype = None
libcd.update_matchcost3.argtypes = [array_3d_float, array_3d_float, c_int, c_int, c_int, array_2d_int, c_int, array_2d_float, c_int]
libcd.update_matchcost4.restype = None
libcd.update_matchcost4.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, array_2d_int, c_int, array_2d_float, c_int]
libcd.update_matchcost5.restype = None
libcd.update_matchcost5.argtypes = [array_3d_float, array_3d_float, array_2d_float, c_int, c_int, c_int, array_2d_int, c_int, array_2d_float, c_int]

def update_matchcost_wrapper2(f1, f2, warppath, metric, dcost):
	nmetric = metric_to_number(metric)
	n, m, c = f1.shape[1], f2.shape[1], f1.shape[2]
	warppath = warppath.astype('int32')
	res = np.zeros((n, n)).astype('float32')
	if dcost == 0:
		libcd.update_matchcost3(f1, f2, n, m, c, warppath, warppath.shape[0], res, nmetric)
	elif dcost == 1:
		z = np.zeros((n, c)).astype('float32')
		libcd.update_matchcost4(f1, f2, z, n, m, c, warppath, warppath.shape[0], res, nmetric)
	elif dcost == 2:
		z = np.zeros((n, c)).astype('float32')
		libcd.update_matchcost5(f1, f2, z, n, m, c, warppath, warppath.shape[0], res, nmetric)
	return res


# libcd.init_product_warping.restype = c_int
# libcd.init_product_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_int, array_2d_double]

# def init_product_warping_wrapper(f1, f2, region):
# 	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
# 	warppath = np.zeros((t1*t2, 2)).astype('int32')
# 	res = np.zeros((n, n)).astype('float64')
# 	region = region.astype('int32')
# 	l = libcd.init_product_warping(f1, f2, t1, t2, n, c, region, warppath, res)
# 	return warppath[:l], res
	


