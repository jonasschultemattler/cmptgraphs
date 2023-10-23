import sys
import time
import os
import argparse

import numpy as np

from dataloader import load_temporal_graphs
from dtgw.vertex_signatures import SIGNATURE_PROVIDERS
# from evaluate import print_dataset_statistics, warping_path_widths

from dtgw.dtgw_ import compute_dtgw
	

def distance_matrix(tadjs, tlabels, graph_labels, signature, init, path, dataset, log=False, window=1000000000):
	print("computing distance matrix with %s and %s..." % (signature, init))
	time_start = time.time()
	if log:
		dmatrix, iterations, wp_distance = compute_distance_matrix(tadjs, tlabels, SIGNATURE_PROVIDERS[signature](), init, log, window)

		wp_widths = warping_path_widths(tadjs, wp_distance)
		print("mean wp width: %.3f" % wp_widths.mean())
		print("max wp width: %.3f" % wp_widths.max())
		print("max abs wp width: %.3f" % wp_distance.max())
		print("mean iterations: %.1f" % iterations.mean())
		print("max iterations: %d" % iterations.max())
	else:
		dmatrix = compute_distance_matrix(tadjs, tlabels, SIGNATURE_PROVIDERS[signature](), init, window=window)

	time_spent = time.time() - time_start
	print("done in %.3fs" % time_spent)

	path = path + "/distances/"
	print("save results in %s" % path)
	if not os.path.exists(path):
		os.mkdir(path)
	name = "_%s_signature__%s_init__window_%d" % (signature, init, window)
	np.savetxt(path + dataset + name + ".distances", np.hstack((dmatrix, graph_labels)))
	with open(path + dataset + name + ".time", 'w+') as time_file:
		time_file.write(str(time_spent*1000) + "\n")

	if log:
		np.savetxt(path + dataset + name + ".iterations", iterations)
		np.savetxt(path + dataset + name + ".wps", wp_distance)



def compute_distance_matrix(tadjs, tlabels, signature_provider, init, log=False, window=1000000000):
	n = len(tadjs)
	features = [signature_provider.signatures(tadjs[i], tlabels[i]) for i in range(n)]

	dmatrix = np.zeros((n, n))
	# if time_limit != float('inf'):
	# 	time_limit = time_limit/((n**2-n)/2)
	if log:
		from dtgw_alternating import compute_dtgw_log
		iterations = np.zeros((n, n))
		wp_distance = np.zeros((n, n))
		for i in range(n):
			print("computing row: %d" % i)
			for j in range(n):
				if j > i:
					# print(j)
					d, iters, wp_d = compute_dtgw_log(features[i], features[j], signature_provider.eps,
						metric=signature_provider.metric, init=init, window=window)
					# d, iters, wp_d = compute_dtgw(features[i], features[j], signature_provider.eps, init=init, window=window, log=True)
					# print(d)
					dmatrix[i,j] = dmatrix[j,i] = d
					iterations[i,j] = iterations[j,i] = iters
					wp_distance[i,j] = wp_distance[j,i] = wp_d
		return dmatrix, iterations, wp_distance
	else:
		for i in range(n):
			print("computing row: %d" % i)
			for j in range(n):
				if j > i:
					d = compute_dtgw(features[i], features[j], signature_provider.eps, init=init, window=window)
					dmatrix[i][j] = dmatrix[j][i] = d
		return dmatrix



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path")
	parser.add_argument("dataset")
	parser.add_argument("--number", type=int, default=10000, help="max number of graphs from dataset to compare")
	parser.add_argument("--signature", choices=SIGNATURE_PROVIDERS.keys(), default="neighbors")
	parser.add_argument("--initialize", 
		choices = ("diagonal_warping", "optimistic_warping", "sigma*", "optimistic_matching"),
		default = "diagonal_warping",
		help = "Initialization to  use for the heuristic"
	)
	parser.add_argument("--time", type=int, default = float('inf'), help="time limit in seconds")
	parser.add_argument("--log", type=bool, default = False, help="log warping path width and iterations")
	parser.add_argument("--window", type=int, default=100, help="abs window size for time warping")

	args = parser.parse_args()

	print("loading temporal graphs...")
	tadjs, tlabels, graph_labels = load_temporal_graphs(args.path + "/" + args.dataset, args.number)
	print_dataset_statistics(tadjs)
	
	distance_matrix(tadjs, tlabels, graph_labels, args.signature, args.initialize, args.path, args.dataset, args.log, args.window)


