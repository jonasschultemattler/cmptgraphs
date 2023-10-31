import sys
import time
import os
import argparse
import functools

from tqdm import tqdm
import numpy as np
import scipy

from multiprocessing import Pool, cpu_count

from matplotlib import pyplot as plt

from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import pairwise


from dataloader import BrainDevelopementDataloader, AbideDataloader, ADHDDataloader

from dtgw.vertex_signatures import SIGNATURE_PROVIDERS
from dtgw.dtgw_ import compute_dtgw, tgw_wrapper, dtw_wrapper, sakoe_chiba_band_wrapper
from dtgw.dtgw import compute_tgw, compute_dtgw_log2


# from tdGraphEmbed.model import TdGraphEmbed
# from tdGraphEmbed.temporal_graph import TemporalGraph
# import graphkernels.kernels as gk
# from grakel.kernels import WeisfeilerLehman
# from grakel import Graph
# import igraph as ig

MAXABSWINDOW = 50

# def node2vec_aligned_kernel(tgraphs, labels, path, dataset):
# 	print("computing tg node2vec embedding kernel...")
# 	time_start = time.time()
# 	kernel = compute_node2vec_aligned_kernel(tgraphs, dataset)
# 	time_spent = time.time() - time_start
# 	print("done in %.3fs" % time_spent)
# 	np.savetxt(os.path.join(path, "node2vec_aligned.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
# 	with open(os.path.join(path, "node2vec_aligned.time"), 'w+') as time_file:
# 		time_file.write(str(time_spent*1000) + "\n")


# def compute_node2vec_aligned_kernel(tgraphs, dataset):
# 	embeddings = []
# 	for tgraph in tqdm(tgraphs):
# 		model = TdGraphEmbed(dataset_name=dataset)
# 		embedding = model.node2vec_aligned(tgraph)
# 		embeddings.append(embedding.flatten())
# 	kernel = pairwise.cosine_similarity(embeddings)
# 	return kernel


# def tgembed_kernel(tgraphs, labels, path, dataset):
# 	print("computing tg embedding kernel...")
# 	time_start = time.time()
# 	kernel = compute_tgembed_kernel(tgraphs, dataset)
# 	time_spent = time.time() - time_start
# 	print("done in %.3fs" % time_spent)
# 	np.savetxt(os.path.join(path, "tgembed.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
# 	with open(os.path.join(path, "tgembed.time"), 'w+') as time_file:
# 		time_file.write(str(time_spent*1000) + "\n")



def compute_tgembed_kernel(tgraphs, dataset):
	embeddings = []
	for tgraph in tqdm(tgraphs):
		model = TdGraphEmbed(dataset_name=dataset)
		documents = model.get_documents_from_graph(tgraph)
		model.run_doc2vec(documents)
		embedding = model.get_embeddings()
		embeddings.append(embedding.flatten())
	kernel = pairwise.cosine_similarity(embeddings)
	return kernel



def corr_kernel(timeseries, labels, path, kind="correlation"):
	print("computing correlation kernel...")
	time_start = time.time()
	connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
	connectomes = connectivity.fit_transform(timeseries)
	kernel = pairwise.linear_kernel(connectomes)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	kernel = pairwise.cosine_similarity(kernel)
	np.savetxt(os.path.join(path, "corr.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	open(os.path.join(path, "corr.time"), 'w').write(str(time_spent*1000) + "\n")


def distance_matrix_tw(timeseries, labels, path, window, metric):
	if metric=="l1":
		metric_func = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")
	elif metric=="l2":
		metric_func = functools.partial(scipy.spatial.distance.cdist, metric="euclidean")
	else:
		return
	print("computing distance matrix...")
	time_start = time.time()
	dmatrix = compute_distance_matrix_tw(timeseries, window, metric_func)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	np.savetxt(os.path.join(path, "tw_%s.distances" % metric), np.hstack((dmatrix, np.array(labels)[:,np.newaxis])))
	with open(os.path.join(path, "tw_%s.time" % metric), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def compute_distance_matrix_tw(timeseries, window, metric):
	n = len(timeseries)
	dmatrix = np.zeros((n, n))
	lifetimes = [ts.shape[0] for ts in timeseries]
	windows = np.zeros((n,n)).astype('int32')
	for i in range(n):
		for j in range(i+1,n):
			windows[i,j] = min(np.ceil(window*max(lifetimes[i], lifetimes[j])), MAXABSWINDOW)
	for i in tqdm(range(n)):
		for j in range(i+1,n):
			cost_matrix = metric(timeseries[i], timeseries[j])
			region = sakoe_chiba_band_wrapper(timeseries[i].shape[0], timeseries[j].shape[0], windows[i,j])
			res, _ = dtw_wrapper(cost_matrix, region)
			dmatrix[i][j] = dmatrix[j][i] = res
	return dmatrix


def WL_tw_kernel(tgraphs, labels, path, window, h):
	print("computing kernel...")
	time_start = time.time()
	kernel = WL_tw(tgraphs, labels, window, h)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	np.savetxt(os.path.join(path, "WL_tw.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	with open(os.path.join(path, "WL_tw.time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def WL_tw(tgraphs, labels, window, h):
	n = len(tgraphs)
	K = np.zeros((n, n))
	graphs = []
	for tgraph in tgraphs:
		for g in tgraph:
			graphs.append(g)
	# graphs = [g for tgraph in tgraphs for g in tgraph]
	# K_WL = gk.CalculateWLKernel(graphs, par=h)
	K_WL = gk.CalculateShortestPathKernel(graphs)
	print(K_WL)
	print(np.histogram(K_WL))
	r = 0
	for i in range(n):
		T_i = len(tgraphs[i])
		c = r+T_i
		for j in range(i+1, n):
			T_j = len(tgraphs[j])
			region = sakoe_chiba_band_wrapper(T_i, T_j, window)
			C = (-K_WL[r:r+T_i,c:c+T_j]).astype('float64')
			res, _ = dtw_wrapper(C, region)
			K[i,j] = K[j,i] = -res
			c += T_j
		r += T_i
	return K

# def WL_tw2(tgraphs, labels, window, h):
# 	n = len(tgraphs)
# 	K = np.zeros((n, n))
# 	# graphs = iter([g for tgraph in tgraphs for g in tgraph])
# 	wl = WeisfeilerLehman(n_iter=h, normalize=True)
# 	# C = wl.fit_transform(graphs)
# 	# print(C)
# 	r = 0
# 	for i in range(n):
# 		T_i = len(tgraphs[i])
# 		c = r+T_i
# 		for j in range(i+1, n):
# 			T_j = len(tgraphs[j])
# 			region = sakoe_chiba_band_wrapper(T_i, T_j, window)
# 			# cost_matrix = (-C[r:r+T_i,c:c+T_j]).astype('float64')
# 			cost_matrix = np.full((T_i, T_j), np.inf)
# 			for k in range(T_i):
# 				for l in range(region[0,k], region[1,k]):
# 					graphs = iter([tgraphs[i][k], tgraphs[j][l]])
# 					K_wl = wl.fit_transform(graphs)
# 					# K_wl = gk.CalculateWLKernel([tgraphs[i][k], tgraphs[j][l]])
# 					# print(K_wl)
# 					cost_matrix[k,l] = -K_wl[0,1]
# 			print(cost_matrix)
# 			res, _ = dtw_wrapper(cost_matrix, region)
# 			K[i,j] = K[j,i] = -res
# 			c += T_j
# 		r += T_i
# 	return K


def RW_tw_kernel(tgraphs, labels, path, window, k):
	print("computing kernel...")
	time_start = time.time()
	kernel = RW_tw(tgraphs, labels, window, k)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	np.savetxt(os.path.join(path, "RW_tw.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	with open(os.path.join(path, "RW_tw.time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def RW_tw(tgraphs, labels, window, k):
	n = len(tgraphs)
	K = np.zeros((n, n))
	graphs = [g for tgraph in tgraphs for g in tgraph]
	# K_RW = gk.CalculateKStepRandomWalkKernel(graphs, par=k)
	K_RW = gk.CalculateKStepRandomWalkKernel(graphs)
	print(np.histogram(K_RW))
	r = 0
	for i in range(n):
		T_i = len(tgraphs[i])
		c = r+T_i
		for j in range(i+1, n):
			T_j = len(tgraphs[j])
			region = sakoe_chiba_band_wrapper(T_i, T_j, window)
			C = (-K_RW[r:r+T_i,c:c+T_j]).astype('float64')
			res, _ = dtw_wrapper(C, region)
			K[i,j] = K[j,i] = -res
			c += T_j
		r += T_i
	return K


def kernel_tadj(tadjs, labels, path):
	print("computing kernel...")
	time_start = time.time()
	X = np.array([tadj.flatten().astype('float64') for tadj in tadjs])
	K = np.dot(X, X.T)
	kernel = pairwise.cosine_similarity(K)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	np.savetxt(os.path.join(path, "tadj.gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	with open(os.path.join(path, "tadj.time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def distance_matrix_tgw(tadjs, labels, path, log, window, signature, k, metric):
	print("computing distance matrix...")
	time_start = time.time()
	dmatrix = compute_distance_matrix_tgw(tadjs, signature, k, metric, window)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	name = "tgw_%s%d_%s_w%s" % (signature, k, metric, ("%.2f" % window).replace('.', ''))
	if metric == "dot":
		kernel = pairwise.cosine_similarity(-dmatrix)
		np.savetxt(os.path.join(path, name + ".gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	else:
		np.savetxt(os.path.join(path, name + ".distances"), np.hstack((dmatrix, np.array(labels)[:,np.newaxis])))
	with open(os.path.join(path, name + ".time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def compute_distance_matrix_tgw(tadjs, signature, k, metric, window):
	n = len(tadjs)
	signature_provider = SIGNATURE_PROVIDERS[signature]()
	features = [signature_provider.signatures(tadjs[i], k) for i in range(n)]
	lifetimes = [tadj.shape[0] for tadj in tadjs]
	windows = np.zeros((n,n)).astype('int32')
	for i in range(n):
		for j in range(i+1,n):
			windows[i,j] = min(np.ceil(window*max(lifetimes[i], lifetimes[j])), MAXABSWINDOW)
	dmatrix = np.zeros((n, n))

	pairs = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
	cpus = cpu_count()
	splits = np.array_split(pairs, cpus)
	with Pool(cpus) as pool:
		kwargs = features, metric, windows
		arg = [(split, kwargs) for split in splits]
		result = pool.map(compute_distance_pairs, arg)
		for split, res in zip(splits, result):
			for index, (i, j) in enumerate(split):
				dmatrix[i,j] = dmatrix[j,i] = res[index]

	return dmatrix


def compute_distance_pairs(arg):
	pairs, kwargs = arg
	features, metric, windows = kwargs
	distances = np.zeros(len(pairs))
	for index, (i, j) in enumerate(pairs):
		d = tgw_wrapper(features[i], features[j], metric, windows[i,j])
		distances[index] = d
	return distances


def distance_matrix_dtgw(tadjs, graph_labels, path, window, signature, k, metric):
	print("computing distance matrix...")
	time_start = time.time()
	dmatrix, matchings = compute_distance_matrix_dtgw(tadjs, signature, k, window, metric)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)

	name = "dtgw_%s%d_%s_w%s" % (signature, k, metric, ("%.2f" % window).replace('.', ''))
	
	np.savetxt(os.path.join(path, name + ".distances"), np.hstack((dmatrix, np.array(graph_labels)[:,np.newaxis])))
	np.savetxt(os.path.join(path, name + ".matchings"), matchings)
	with open(os.path.join(path, name + ".time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")



def compute_distance_matrix_dtgw(tadjs, signature, k, window, metric, init="diagonal_warping"):
	signature_provider = SIGNATURE_PROVIDERS[signature]()
	features = [signature_provider.signatures(tadj, k) for tadj in tadjs]
	n = len(tadjs)
	dmatrix = np.zeros((n, n)).astype('float32')
	matched = np.zeros((n, n))

	lifetimes = [tadj.shape[0] for tadj in tadjs]
	windows = np.zeros((n,n)).astype('int32')
	for i in range(n):
		for j in range(i+1,n):
			windows[i,j] = min(np.ceil(window*max(lifetimes[i], lifetimes[j])), MAXABSWINDOW)

	print("compute dtgw-distances...")
	pairs = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
	cpus = cpu_count()
	splits = np.array_split(pairs, cpus)
	with Pool(cpus) as pool:
		kwargs = features, metric, windows
		arg = [(split, kwargs) for split in splits]
		results = pool.map(compute_distance_pairs2, arg)
		for split, res in zip(splits, results):
			for index, (i, j) in enumerate(split):
				d, m = res[0][index], res[1][index]
				dmatrix[i,j] = dmatrix[j,i] = d
				matched[i,j] = matched[j,i] = m
		return dmatrix, matched


def compute_distance_pairs2(arg):
	pairs, kwargs = arg
	features, metric, windows = kwargs
	distances = np.zeros(len(pairs))
	matched = np.zeros(len(pairs))
	for index, (i, j) in enumerate(pairs):
		d, m = compute_dtgw_log2(features[i], features[j], 0, metric=metric, window=windows[i,j])
		distances[index], matched[index] = d, m
	return distances, matched



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset")
	parser.add_argument("--savetxt", type=bool, default=False)
	parser.add_argument("--distinct_labels", type=bool, default=True)
	parser.add_argument("--roi", choices=("atlas",), default="atlas")
	parser.add_argument("--number", type=int, default=100, help="max number of graphs from dataset to compare")
	parser.add_argument("--log", type=bool, default=False, help="log warping path width and iterations")
	parser.add_argument("--window", type=float, default=0.2, help="rel window size for time warping")
	parser.add_argument("--alg", choices=("tw", "corr", "adj", "dtgw", "tgw", "tgembed", "n2v_aligned", "wl_tw", "rw_tw"), default="dtgw", help="algorithm to compute distances")
	parser.add_argument("--signature", choices=("degree", "interaction"), default="interaction", help="vertex signature")
	parser.add_argument("--k", type=int, default=2, help="k neighborhood")
	parser.add_argument("--metric", choices=("l1", "l2", "dot"), default="l1", help="metric norm")
	parser.add_argument("--h", type=int, default=2, help="WL depth")
	args = parser.parse_args()

	dataset_path = os.path.join("..", "datasets", "brains")
	output_path = os.path.join("..", "output", "brains", args.dataset)
	if not os.path.exists(output_path):
		os.mkdir(output_path)

	if args.dataset == "development":
		dataloader = BrainDevelopementDataloader(args.dataset, dataset_path, args.number, args.roi)
	elif args.dataset == "abide":
		dataloader = AbideDataloader(args.dataset, dataset_path, args.number, args.roi)
	elif args.dataset == "adhd":
		dataloader = ADHDDataloader(args.dataset, dataset_path, args.number, args.roi)
	else:
		pass

	if args.savetxt:
		dataloader.savetxt(args.distinct_labels)

	if args.alg == "corr":
		X, y = dataloader.get_timeseries_data()
		kernel = corr_kernel(X, y, output_path)
	elif args.alg == "tw":
		X, y = dataloader.get_timeseries_data()
		distance_matrix_tw(X, y, output_path, args.window, args.metric)
	elif args.alg == "adj":
		X, y = dataloader.get_data()
		kernel_tadj(X, y, output_path)
	elif args.alg == "tgw":
		X, y = dataloader.get_data()
		distance_matrix_tgw(X, y, output_path, args.log, args.window, args.signature, args.k, args.metric)
	elif args.alg == "dtgw":
		X, y = dataloader.get_data()
		distance_matrix_dtgw(X, y, output_path, args.window, args.signature, args.k, args.metric)
	elif args.alg == "wl_tw":
		X, y = dataloader.get_ig_data()
		# X, y = dataloader.get_gk_data()
		WL_tw_kernel(X, y, output_path, args.window, args.h)
	elif args.alg == "rw_tw":
		X, y = dataloader.get_ig_data()
		RW_tw_kernel(X, y, output_path, args.window, args.h)
	# elif args.alg == "tgembed":
	# 	X, y = dataloader.get_nx_data()
	# 	tgembed_kernel(X, y, output_path, args.dataset)
	# elif args.alg == "n2v_aligned":
	# 	X, y = dataloader.get_nx_data()
	# 	node2vec_aligned_kernel(X, y, output_path, args.dataset)
	else:
		pass


