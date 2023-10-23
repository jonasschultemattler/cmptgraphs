import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double


array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype='int32', ndim=2, flags='CONTIGUOUS')
array_2d_bool = npct.ndpointer(dtype=bool, ndim=2, flags='CONTIGUOUS')
array_3d_bool = npct.ndpointer(dtype=bool, ndim=3, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("warping_test.so", "dtgw/build")

# libcd.metric.restype = None
# libcd.metric.argtypes = [array_2d_double, array_2d_double, c_int, c_int, c_int, array_2d_double]


# libcd.dtgw.restype = c_double
# libcd.dtgw.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

# def dtgw_wrapper(f1, f2, init, window, max_iterations):
# 	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
# 	if init == "diagonal_warping":
# 		i = 1
# 	elif init == "optimistic_warping":
# 		i = 2
# 	elif init == "optimistic_matching":
# 		i = 3
# 	elif init == "sigma*":
# 		i = 4
# 	else:
# 		raise ValueError("Invalid initialization")
# 	return libcd.dtgw(f1, f2, t1, t2, n, c, i, window, max_iterations)


libcd.update_warping.restype = None
libcd.update_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, array_1d_int, c_int, array_2d_int, array_2d_double]

def warping_update_wrapper(f1, f2, matching, region):
	t1, t2, c = f1.shape[0], f2.shape[0], f1.shape[2]
	res = np.full((t1, t2), np.inf).astype('float64')
	f1 = f1.astype('float64')
	f2 = f2.astype('float64')
	region = region.astype('int32')
	matching = matching.astype('int32')
	libcd.update_warping(f1, f2, t1, t2, c, matching, matching.shape[0], region, res)
	return res

# def metric(f1, f2):
# 	t1, t2, c = f1.shape[0], f2.shape[0], f1.shape[1]
# 	res = np.zeros((t1, t2)).astype('float64')
# 	f1 = f1.astype('float64')
# 	f2 = f2.astype('float64')
# 	libcd.metric(f1, f2, t1, t2, c, res)
# 	return res

# libcd.lapjv.restype = c_double
# libcd.lapjv.argtypes = [c_int, array_2d_double, array_1d_int, array_1d_int]

# def lap_wrapper(assigncost):
# 	n = assigncost.shape[0]
# 	rowsol = np.zeros(n).astype('int32')
# 	colsol = np.zeros(n).astype('int32')
# 	assigncost = assigncost.astype('float64')
# 	cost = libcd.lapjv(n, assigncost, rowsol, colsol)
# 	return cost, rowsol[colsol]

libcd.update_matchcost.restype = None
libcd.update_matchcost.argtypes = [array_3d_double, array_3d_double, c_int, c_int, array_2d_int, c_int, array_2d_double]

def update_matchcost_wrapper(f1, f2, warppath):
	n, c = f1.shape[1], f1.shape[2]
	warppath = warppath.astype('int32')
	res = np.zeros((n, n)).astype('float64')
	libcd.update_matchcost(f1, f2, n, c, warppath, warppath.shape[0], res)
	return res

libcd.dynamic_timewarping.restype = c_int
libcd.dynamic_timewarping.argtypes = [array_2d_double, c_int, c_int, array_2d_int, array_2d_int, array_2d_double]

def dtw_wrapper(cost_matrix, region):
	t1, t2 = cost_matrix.shape
	res = np.full((t1+1, t2+1), np.inf).astype('float64')
	region = region.astype('int32')
	path = np.zeros((t1*t2, 2), dtype=int).astype('int32')
	l = libcd.dynamic_timewarping(cost_matrix, t1, t2, region, path, res)
	path = path[:l,:]
	return res[0,0], path



libcd.init_product_warping.restype = c_int
libcd.init_product_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_int, array_2d_double]

def init_product_warping_wrapper(f1, f2, region):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	warppath = np.zeros((t1*t2, 2)).astype('int32')
	res = np.zeros((n, n)).astype('float64')
	region = region.astype('int32')
	l = libcd.init_product_warping(f1, f2, t1, t2, n, c, region, warppath, res)
	return warppath[:l], res

libcd.init_opt_matching.restype = None
libcd.init_opt_matching.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_double]

def init_opt_matching_wrapper(f1, f2, region):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	res = np.zeros((n, n)).astype('float64')
	region = region.astype('int32')
	libcd.init_opt_matching(f1, f2, t1, t2, n, c, region, res)
	return res

libcd.init_diagonal_warping.restype = c_int
libcd.init_diagonal_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_double]

def init_diagonal_warping_wrapper(f1, f2):
	t1, t2 = f1.shape[0], f2.shape[0]
	n, c = f1.shape[1], f1.shape[2]
	path = np.zeros((t1+t2, 2)).astype('int32')
	res = np.zeros((n, n)).astype('float64')
	l = libcd.init_diagonal_warping(f1, f2, t1, t2, n, c, path, res)
	return res, path[:l,:]

libcd.init_opt_warping.restype = None
# libcd.init_opt_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int,
# array_2d_int, array_2d_double, array_2d_double, array_1d_int, array_1d_int, array_1d_double, array_1d_double]
libcd.init_opt_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_double]

def init_opt_warping_wrapper(f1, f2, region):
	t1, t2 = f1.shape[0], f2.shape[0]
	n, c = f1.shape[1], f1.shape[2]
	region = region.astype('int32')
	# tmp = np.zeros((n, n)).astype('float64')
	res = np.empty((t1, t2)).astype('float64')
	# rowsol = np.zeros(n).astype('int32')
	# colsol = np.zeros(n).astype('int32')
	# rowcost = np.zeros(n).astype('float64')
	# colcost = np.zeros(n).astype('float64')
	libcd.init_opt_warping(f1, f2, t1, t2, n, c, region, res)
	return res


libcd.vertex_feature_labels.restype = None
libcd.vertex_feature_labels.argtypes = [array_2d_bool, c_int, c_int, array_3d_double]

def vertex_feature_labels_wrapper(tlabels):
	t, n = tlabels.shape
	feature = np.zeros((t, n, 1)).astype('float64')
	libcd.vertex_feature_labels(tlabels, n, t, feature)
	return feature


libcd.vertex_feature_neighbors.restype = None
libcd.vertex_feature_neighbors.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_double]

def vertex_feature_neighbors_wrapper(tadj, tlabels):
	t, n = tlabels.shape
	feature = np.zeros((t, n, 3)).astype('float64')
	libcd.vertex_feature_neighbors(tadj, tlabels, n, t, feature)
	return feature

libcd.vertex_feature_neighbors_normed.restype = None
libcd.vertex_feature_neighbors_normed.argtypes = [array_3d_bool, array_2d_bool, c_int, c_int, array_3d_double]

def vertex_feature_neighbors_normed_wrapper(tadj, tlabels):
	t, n = tlabels.shape
	feature = np.zeros((t, n, 3)).astype('float64')
	libcd.vertex_feature_neighbors_normed(tadj, tlabels, n, t, feature)
	return feature

libcd.sakoe_chiba_band.restype = None
libcd.sakoe_chiba_band.argtypes = [c_int, c_int, c_int, array_2d_int]

def sakoe_chiba_band_wrapper(t1, t2, window):
	region = np.zeros((2, t1), dtype=int).astype('int32')
	libcd.sakoe_chiba_band(t1, t2, window, region)
	return region

