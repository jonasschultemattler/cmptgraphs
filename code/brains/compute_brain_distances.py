import sys
import time
import os
import argparse
from tqdm import tqdm
import numpy as np
import scipy
import functools

# from brain_dataloader import load_temporal_graphs, load_timeseries, load_labels
from vertex_signatures import SIGNATURE_PROVIDERS
from evaluate import print_dataset_statistics, warping_path_widths

from dtgw_ import compute_dtgw
from dtgw_alternating import compute_tgw


def norm_matrix(matrix):
    nmatrix = np.empty(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            d = np.sqrt(matrix[i, i]) * np.sqrt(matrix[j, j]);
            if d != 0:
                nmatrix[i, j] = matrix[i, j]/d;
            else:
                nmatrix[i, j] = 0;
    return nmatrix


def corr_kernel(data, kind="correlation"):
    connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
    connectomes = connectivity.fit_transform(data)
    kernel = pairwise.linear_kernel(connectomes)
    return kernel


def distance_matrix_tw(timeseries, labels, path, dataset, window, metric):
	print("computing distance matrix...")
	time_start = time.time()
	dmatrix = compute_distance_matrix_tw(timeseries, window, metric)
	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	path = os.path.join(path, "distances")
	if not os.path.exists(path):
		os.mkdir(path)
	print("save results in %s" % path)
	name = dataset + "_tw"
	print(labels)
	print(labels.shape)
	np.savetxt(os.path.join(path, name + ".distances"), np.hstack((dmatrix, labels)))
	with open(os.path.join(path, name + ".time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")


def distance_matrix(tadjs, graph_labels, path, dataset, log, window, metric, signature="degree", algorithm="dtgw"):
	print("computing distance matrix...")
	time_start = time.time()
	if algorithm == "dtgw":
		if log:
			dmatrix, iterations, wp_distance = compute_distance_matrix_dtgw(tadjs, SIGNATURE_PROVIDERS[signature](), log, window)

			wp_widths = warping_path_widths(tadjs, wp_distance)
			print("mean wp width: %.3f" % wp_widths.mean())
			print("max wp width: %.3f" % wp_widths.max())
			print("max abs wp width: %.3f" % wp_distance.max())
			print("mean iterations: %.1f" % iterations.mean())
			print("max iterations: %d" % iterations.max())
		else:
			dmatrix = compute_distance_matrix_dtgw(tadjs, SIGNATURE_PROVIDERS[signature](), log, window)
	if algorithm == "tgw":
		if log:
			return
		else:
			dmatrix = compute_distance_matrix_tgw(tadjs, SIGNATURE_PROVIDERS[signature](), window)

	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)
	print(dmatrix)
	path = os.path.join(path, "distances")
	if not os.path.exists(path):
		os.mkdir(path)
	# path = os.path.join(path, dataset)
	# if not os.path.exists(path):
	# 	os.mkdir(path)
	print("save results in %s" % path)
	name = dataset + "_" + algorithm + "_" + signature
	np.savetxt(os.path.join(path, name + ".distances"), np.hstack((dmatrix, graph_labels)))
	with open(os.path.join(path, name + ".time"), 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")
	if log:
		np.savetxt(os.path.join(path, name + ".iterations"), iterations)
		np.savetxt(os.path.join(path, name + ".wps"), wp_distance)



def compute_distance_matrix_dtgw(tadjs, signature_provider, log, window, init="diagonal_warping"):
	n = len(tadjs)
	features = [signature_provider.signatures(tadjs[i]) for i in range(n)]

	dmatrix = np.zeros((n, n))
	if log:
		from dtgw_alternating import compute_dtgw_log
		iterations = np.zeros((n, n))
		wp_distance = np.zeros((n, n))
		for i in tqdm(range(n)):
			# print("computing row: %d" % i)
			for j in range(n):
				if j > i:
					d, iters, wp_d = compute_dtgw_log(features[i], features[j], signature_provider.eps,
						metric=signature_provider.metric, init=init, window=window)
					dmatrix[i,j] = dmatrix[j,i] = d
					iterations[i,j] = iterations[j,i] = iters
					wp_distance[i,j] = wp_distance[j,i] = wp_d
		return dmatrix, iterations, wp_distance
	else:
		for i in range(n):
			print("computing row: %d" % i)
			for j in range(n):
				if j > i:
					d = compute_dtgw(features[i], features[j], signature_provider.eps, metric=signature_provider.metric, init=init, window=window)
					dmatrix[i][j] = dmatrix[j][i] = d
		return dmatrix


def compute_distance_matrix_tgw(tadjs, signature_provider, window):
	n = len(tadjs)
	features = [signature_provider.signatures(tadjs[i]) for i in range(n)]

	dmatrix = np.zeros((n, n))
	for i in tqdm(range(n)):
		for j in range(n):
			if j > i:
				d = compute_tgw(features[i], features[j], signature_provider.eps, metric=signature_provider.metric, window=window)
				dmatrix[i][j] = dmatrix[j][i] = d
	return dmatrix


def compute_distance_matrix_tw(timeseries, window):
	n = len(timeseries)
	dmatrix = np.zeros((n, n))
	for i in tqdm(range(n)):
		for j in range(n):
			if j > i:
				d = compute_tgw(timeseries[i], timeseries[j], 0, window=window)
				dmatrix[i][j] = dmatrix[j][i] = d
	return dmatrix



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path")
	parser.add_argument("dataset")
	parser.add_argument("--roi", choices=("atlas"))
	parser.add_argument("--number", type=int, default=10000, help="max number of graphs from dataset to compare")
	parser.add_argument("--log", type=bool, default=False, help="log warping path width and iterations")
	parser.add_argument("--window", type=int, default=100, help="abs window size for time warping")
	parser.add_argument("--alg", choices=("dtgw", "tgw", "tw"), default="dtgw", help="algorithm to compute distances")
	parser.add_argument("--signature", choices=("degree"), default="degree", help="vertex signature")
	parser.add_argument("--metric", choices=("l1", "l2"), default="l1", help="metric norm")

	args = parser.parse_args()
	# if args.metric == "l1":
	# 	metric = scipy.spatial.distance.cdist
	# if args.metric == "l2":
	# 	metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

	if args.alg == "tw":
		timeseries_dict = load_timeseries(args.path, args.dataset, n=args.number)
		timeseries = [(data, subject) for subject, data in timeseries_dict.items()]
		graph_labels = load_labels(args.path, args.dataset, [ts[1] for ts in timeseries])
		timeseries_data = [ts[0][:,:,np.newaxis] for ts in timeseries]
		distance_matrix_tw(timeseries_data, graph_labels, args.path, args.dataset, window=args.window)
	else:
		print("loading temporal graphs...")
		tadjs, graph_labels = load_temporal_graphs(args.path, args.dataset, args.roi, n=args.number)
		# print_dataset_statistics(tadjs)
		distance_matrix(tadjs, graph_labels, args.path, args.dataset, args.log, args.window, signature=args.signature, algorithm=args.alg, metric=metric)


