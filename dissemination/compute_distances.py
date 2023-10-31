import time
import os
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from sklearn.metrics import pairwise

from dataloader import DisseminationDataloader

from dtgw.vertex_signatures import SIGNATURE_PROVIDERS
from dtgw.dtgw_ import compute_dtgw
from dtgw.dtgw import compute_dtgw_log



MAXABSWINDOW = 100



def compute_distance_matrix(tgraphs, signature_provider, k, dcost, init, metric, log, window, max_iterations):
	print("compute vertex features...")
	features = [signature_provider.signatures(tgraph.tadj, tgraph.tlabels, k) for tgraph in tgraphs]
	print("done")
	n = len(tgraphs)
	dmatrix = np.zeros((n, n)).astype('float32')
	iterations = np.zeros((n, n))
	wp_distance = np.zeros((n, n))

	lifetimes = [tgraph.tadj.shape[0] for tgraph in tgraphs]
	windows = np.zeros((n,n)).astype('int32')
	for i in range(n):
		for j in range(i+1,n):
			windows[i,j] = min(np.ceil(window*max(lifetimes[i], lifetimes[j])), MAXABSWINDOW)

	print("compute dtgw-distances...")
	pairs = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
	cpus = cpu_count()
	splits = np.array_split(pairs, cpus)
	with Pool(cpus) as pool:
		kwargs = features, dcost, metric, init, windows, max_iterations
		arg = [(split, kwargs) for split in splits]
		results = pool.map(compute_distance_pairs, arg)
		for split, res in zip(splits, results):
			for index, (i, j) in enumerate(split):
				d, iters, wp = res[0][index], res[1][index], res[2][index]
				dmatrix[i,j] = dmatrix[j,i] = d
				iterations[i,j] = iterations[j,i] = iters
				wp_distance[i,j] = wp_distance[j,i] = wp
		return dmatrix, iterations, wp_distance



def compute_distance_pairs(arg):
	pairs, kwargs = arg
	features, dcost, metric, init, windows, max_iterations = kwargs
	distances = np.zeros(len(pairs))
	iterations = np.zeros(len(pairs))
	wps = np.zeros(len(pairs))
	for index, (i, j) in enumerate(pairs):
		d, iters, wp = compute_dtgw_log(features[i], features[j], dcost, metric=metric, init=init, window=windows[i,j], max_iterations=max_iterations)
		distances[index], iterations[index], wps[index] = d, iters, wp
	return distances, iterations, wps



def distance_matrix(dataset, tgraphs, labels, signature, k, dcost, init, metric, window, max_iterations, log, output_path):
	n = len(tgraphs)
	signature_provider = SIGNATURE_PROVIDERS[signature]()
	name = "dtgw_%s%d_c%d_%s_%s_w%s_i%d_n%d" % (signature, k, dcost, init, metric, ("%.2f" % window).replace('.', ''), max_iterations, n)
	print("computing distance matrix %s ..." % name)
	time_start = time.time()
	dmatrix, iterations, wp_distance = compute_distance_matrix(tgraphs, signature_provider, k, dcost, init, metric, log, window, max_iterations)
	time_spent = time.time() - time_start

	print("done in %.3fs" % time_spent)
	
	save_results(output_path, dataset, name, metric, dmatrix, time_spent, iterations, wp_distance)
	


def save_results(output_path, dataset, name, metric, dmatrix, time, iterations=None, wp_distance=None, normalize=True):
	path = os.path.join(output_path, dataset)
	if not os.path.exists(path):
		os.mkdir(path)
	print("save results in %s" % path)
	if metric == "dot":
		kernel = -dmatrix
		if normalize:
			kernel = pairwise.cosine_similarity(kernel)
		np.savetxt(os.path.join(path, name + ".gram"), np.hstack((kernel, np.array(labels)[:,np.newaxis])))
	else:
		if normalize:
			dmatrix /= np.max(dmatrix)
		np.savetxt(os.path.join(path, name + ".distances"), np.hstack((dmatrix, np.array(labels)[:,np.newaxis])))

	with open(os.path.join(path, name + ".time"), 'w+') as time_file:
		time_file.write(str(time*1000) + "\n")
	if iterations is not None:
		np.savetxt(os.path.join(path, name + ".iterations"), iterations)
	if wp_distance is not None:
		np.savetxt(os.path.join(path, name + ".wps"), wp_distance)
		


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("path")
	parser.add_argument("dataset")
	parser.add_argument("--number", type=int, default=100, help="max number of graphs from dataset to compare")
	parser.add_argument("--signature", choices=SIGNATURE_PROVIDERS.keys(), default="subtrees")
	parser.add_argument("--k", type=int, default=2, help="k depth subtree; k length random walks")
	parser.add_argument("--dcost", type=int, default=0, help="pay 0, f(v) or f(v)/|n-m| for not matched vertices")
	parser.add_argument("--init",
		choices = ("diagonal_warping", "optimistic_warping", "sigma*", "optimistic_matching"),
		default = "diagonal_warping", help = "Initialization to  use for the heuristic")
	parser.add_argument("--metric", choices=["l1", "l2", "dot"], default="l1")
	# parser.add_argument("--window", type=int, default=5, help="abs window size for time warping")
	parser.add_argument("--window", type=float, default=0.2, help="rel window size for time warping")
	parser.add_argument("--iterations", type=int, default=5, help="max iterations per heuristic computation")
	parser.add_argument("--log", type=bool, default=True, help="log warping path width and iterations")
	args = parser.parse_args()

	path = "../datasets/dissemination"
	output_path = "../output/dissemination"

	dataloader = DisseminationDataloader(args.dataset, path)
	tgraphs, labels = dataloader.loadtxt(args.number)

	distance_matrix(args.dataset, tgraphs, labels, args.signature, args.k, args.dcost, args.init, args.metric, args.window, args.iterations, args.log, output_path)



