import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_void_p
from tools import shortest_warp_path2
import itertools
import scipy.spatial

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype='int32', ndim=2, flags='CONTIGUOUS')
array_2d_bool = npct.ndpointer(dtype=bool, ndim=2, flags='CONTIGUOUS')
array_3d_bool = npct.ndpointer(dtype=bool, ndim=3, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("warping_test", ".")


# libcd.warping_update.restype = None
# libcd.warping_update.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, array_1d_int, c_int, array_2d_int, array_2d_double]

# def enter(f1, f2, matching):
# 	t1, t2, n = f1.shape[0], f2.shape[0], f1.shape[2]
# 	res = np.full((t1, t2), np.inf)
# 	region = sakoe_chiba_band(t1, t2, 3).astype('int32')
# 	libcd.warping_update(f1, f2, t1, t2, n, matching, len(matching), region, res)
# 	return res


# libcd.metric.restype = None
# libcd.metric.argtypes = [array_2d_double, array_2d_double, c_int, c_int, c_int, array_2d_double]

# def metric(f1, f2):
# 	t1, t2, c = f1.shape[0], f2.shape[0], f1.shape[1]
# 	res = np.zeros((t1, t2)).astype('float64')
# 	f1 = f1.astype('float64')
# 	f2 = f2.astype('float64')
# 	libcd.metric(f1, f2, t1, t2, c, res)
# 	return res


libcd.update_matchcost.restype = None
libcd.update_matchcost.argtypes = [array_3d_double, array_3d_double, c_int, c_int, array_2d_int, c_int, array_2d_double]

def update_matchcost(f1, f2, warppath):
	n, c = f1.shape[1], f1.shape[2]
	warppath = warppath.astype('int32')
	res = np.zeros((n, n)).astype('float64')
	libcd.update_matchcost(f1, f2, n, c, warppath, warppath.shape[0], res)
	return res

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
libcd.init_opt_matching.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_double, array_2d_double]

def init_opt_matching_wrapper(f1, f2, region):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	res = np.zeros((n, n)).astype('float64')
	tmp = np.full((t1, t2), np.inf).astype('float64')
	region = region.astype('int32')
	libcd.init_opt_matching(f1, f2, t1, t2, n, c, region, tmp, res)
	return res

libcd.init_opt_warping.restype = None
libcd.init_opt_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int,
array_2d_double, array_2d_double, array_1d_int, array_1d_int, array_1d_double, array_1d_double]

def init_opt_warping(f1, f2, region):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	tmp = np.zeros((n, n)).astype('float64')
	res = np.zeros((t1, t2)).astype('float64')
	region = region.astype('int32')
	rowsol = np.zeros(n).astype('int32')
	rowcost = np.zeros(n).astype('float64')
	colsol = np.zeros(n).astype('int32')
	colcost = np.zeros(n).astype('float64')
	libcd.init_opt_warping(f1, f2, t1, t2, n, c, region, tmp, res, rowsol, colsol, rowcost, colcost)
	return res

libcd.shortest_warp_path.restype = c_int
libcd.shortest_warp_path.argtypes = [c_int, c_int, array_2d_int]

def shortest_warp_path_wrapper(n, m):
	path = np.zeros((n*m, 2)).astype('int32')
	l = libcd.shortest_warp_path(n, m, path)
	return path[:l,:]


libcd.init_diagonal_warping.restype = c_int
libcd.init_diagonal_warping.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, array_2d_int, array_2d_double]

def init_diagonal_warping_wrapper(f1, f2):
	t1, t2 = f1.shape[0], f2.shape[0]
	n, c = f1.shape[1], f1.shape[2]
	path = np.zeros((t1+t2, 2)).astype('int32')
	res = np.zeros((n, n)).astype('float64')
	l = libcd.init_diagonal_warping(f1, f2, t1, t2, n, c, path, res)
	return res, path[:l,:]


# f1 = np.asarray(np.arange(50).reshape(10,5,1), dtype=float)
# f2 = np.asarray(np.arange(60).reshape(12,5,1), dtype=float)
# matching = np.arange(5).astype('int32')
# res = enter(f1, f2, matching)
# print(f1)
# print(f2)
# print(res)
# print(res.shape)

# f1 = np.asarray(np.arange(100).reshape(10,5,2), dtype=float)
# f2 = np.asarray(np.arange(120).reshape(12,5,2), dtype=float)

# res = metric(f1[:,2,:], f2[:,4,:])
# print(res)
# res = metric(f1[5,:,:], f2[7,:,:])
# print(res)

# f1 = np.asarray(np.arange(100).reshape(10,5,2), dtype=float)
# f2 = np.asarray(np.arange(120).reshape(12,5,2), dtype=float)

# res = metric(f1[:,2,:], f2[:,4,:])
# print(res)
# res = metric(f1[5,:,:], f2[7,:,:])
# print(res)

# f1 = np.asarray(np.arange(30).reshape(6,5,1), dtype=float)
# f2 = np.asarray(np.arange(25).reshape(5,5,1), dtype=float)
# t1, t2 = f1.shape[0], f2.shape[0]
# n = f1.shape[1]
# region = sakoe_chiba_band(t1, t2, 2)

# print(f1)
# print(f2)
# print(region)

# res = init_opt_warping(f1, f2, region)
# print(res)


# res, path = init_diagonal_warping_wrapper(f1, f2)
# print(res)
# print(path)
# matchcost = update_matchcost(f1, f2, warppath)
# print(matchcost)

libcd.vertex_feature_labels.restype = None
libcd.vertex_feature_labels.argtypes = [array_2d_bool, c_int, c_int, array_3d_double]

def vertex_feature_labels_wrapper(tadj, tlabels):
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


# def tgraph_to_matrix(tg):
# 	# boolean tgraph t x n x n
# 	# boolean labels t x n
# 	n = tg.tgraph.num_vertices()
# 	T = tg.tgraph.lifetime
# 	tadj = np.zeros((T,n,n), dtype=bool)
# 	for t, v, w in tg.tgraph.timeedges():
# 		tadj[t,v,w] = tadj[t,w,v] = True
# 	return tadj, tg.tgraph.vertex_labels


# from dataloader import load_temporal_graphs
# path = "../../../../datasets/"
# dataset = "infectious_ct1"
# from vertex_signatures import LabelSignatureProvider, NeighborhoodSignatureProvider, NormedNeighborhoodSignatureProvider

# tgraphs = load_temporal_graphs(path + dataset + "/" + dataset)
# features = []
# for tgraph in tgraphs:
# 	tadj, tlabels = tgraph_to_matrix(tgraph)
# 	f = vertex_feature_neighbors_normed_wrapper(tadj, tlabels)
# 	# f = vertex_feature_labels_wrapper(tadj, tlabels)
# 	# fl = LabelSignatureProvider().signatures(tgraph)
# 	# f2 = NeighborhoodSignatureProvider().signatures(tgraph)
# 	features.append(f)


libcd.sakoe_chiba_band.restype = None
libcd.sakoe_chiba_band.argtypes = [c_int, c_int, c_int, array_2d_int]

def sakoe_chiba_band_wrapper(t1, t2, window):
	region = np.zeros((2, t1), dtype=int).astype('int32')
	libcd.sakoe_chiba_band(t1, t2, window, region)
	# region = np.clip(region[:, :t1], 0, t2)
	return region


region = sakoe_chiba_band_wrapper(5, 5, 2)
print(region)


