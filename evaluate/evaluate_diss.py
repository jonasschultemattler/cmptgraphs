import os
import csv
import json
import numpy as np

from dissemination.dataloader import DisseminationDataloader, load_temporal_graphs


def print_dataset_statistics(tadjs):
	n = len(tadjs)
	v, e, t = np.empty(n), np.empty(n), np.empty(n)
	for i, tadj in enumerate(tadjs):
		v[i] = tadj.shape[1]
		e[i] = np.sum(tadj)/2 # len(list(tg.tgraph.timeedges()))
		t[i] = tadj.shape[0]
	print("%d temporal graphs" % n)
	print("vertices: max: %d mean: %.1f" % (v.max(), v.mean()))
	print("edges: max: %d mean: %.1f" % (e.max(), e.mean()))
	print("lifetime: max: %d mean: %.1f" % (t.max(), t.mean()))


def warping_path_widths(tadjs, wp_distance):
	num_vertices = [tadj.shape[1] for tadj in tadjs]
	max_vertices = np.empty(wp_distance.shape)
	for i in range(len(tadjs)):
		for j in range(len(tadjs)):
			max_vertices[i,j] = max(tadjs[i].shape[1], tadjs[j].shape[1])
	wp_widths = wp_distance/max_vertices
	return wp_widths


def running_times(path):
	datasets = [directory for directory in os.listdir(path) if os.path.exists(path + directory + "/distances")]
	configs = set()
	for dataset in datasets:
		n = len(dataset) + 1
		for file_name in os.listdir(path + dataset + "/distances/"):
			if file_name.endswith("5.time"):
				config = file_name[n:-5]
				configs.add(config)
	configs = sorted(list(configs))
	data = -1*np.ones((len(configs), len(datasets)))
	for i, dataset in enumerate(datasets):
		for j, config in enumerate(configs):
			if os.path.exists(path + dataset + "/distances/" + dataset + "_" + config + ".time"):
				file = open(path + dataset + "/distances/" + dataset + "_" + config + ".time", "r")
				data[j,i] = float(file.read()[:-2])/1000
	translate = {"optimistic_matching": "$\\sigma_{\\text{opt}}$",
	"sigma*": "$\\sigma^*$", "diagonal_warping": "swp", "optimistic_warping": "owp"}
	with open(path + "times_window.csv", "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["signature", "init"] + datasets)
		for j, config in enumerate(configs):
			# signature, init = config.split('__')
			signature, init, _ = config.split('__')
			writer.writerow([signature[:-10], translate[init[:-5]]] + list(data[j,:]))


def statistics_datasets(path):
	datasets = [directory for directory in os.listdir(path) if os.path.exists(path + directory + "/distances")]
	data = np.empty((7, len(datasets)))
	for j, dataset in enumerate(datasets):
		tadjs, all_node_labels, graph_labels = load_temporal_graphs(path + dataset + "/" + dataset)
		n = len(tadjs)
		v, e, t = np.empty(n), np.empty(n), np.empty(n)
		for i, tadj in enumerate(tadjs):
			t[i] = tadj.shape[0]
			v[i] = tadj.shape[1]
			e[i] = np.sum(tadj)/2
		data[0,j] = n
		data[1,j] = v.max()
		data[2,j] = v.mean()
		data[3,j] = e.max()
		data[4,j] = e.mean()
		data[5,j] = t.max()
		data[6,j] = t.mean()
	with open(path + "datasets.csv", "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["properties"] + datasets)
		writer.writerow(["$|G|$"] + list(data[0,:]))
		writer.writerow(["$\\max |V|$"] + list(data[1,:]))
		writer.writerow(["$\\varnothing |V|$"] + list(data[2,:]))
		writer.writerow(["$\\max |E|$"] + list(data[3,:]))
		writer.writerow(["$\\varnothing |E|$"] + list(data[4,:]))
		writer.writerow(["$\\max t$"] + list(data[5,:]))
		writer.writerow(["$\\varnothing t$"] + list(data[6,:]))


def running_times_init(path, datasets, signatures, inits, metrics):
	metric = "l1"
	window = 5
	data = -1*np.ones((len(signatures)*len(inits), len(datasets)))
	for d, dataset in enumerate(datasets):
		for s, signature in enumerate(signatures):
			for i, init in enumerate(inits):
				name = "dtgw_%s_%s_%s_w%d" % (signature, init, metric, window)
				matrix = np.loadtxt(os.path.join(path, dataset, name + ".distances"))
				n = matrix.shape[0]
				with open(os.path.join(path, dataset, name + ".time"), "r") as file:
					data[s*len(inits)+i,d] = float(file.read()[:-2])/((n*n-n)/2)
	translate = {"optimistic_matching": "$\\sigma_{\\text{opt}}$",
	"sigma*": "$\\sigma^*$", "diagonal_warping": "swp", "optimistic_warping": "owp"}
	with open(os.path.join(path, "running_times_heuristic_signature_init.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["signature", "init"] + datasets)
		for s, signature in enumerate(signatures):
			for i, init in enumerate(inits):
				writer.writerow([signature, translate[init]] + list(data[s*len(inits)+i,:]))
	with open(os.path.join(path, "running_times_heuristic_init.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["init"] + datasets)
		for i, init in enumerate(inits):
			writer.writerow([translate[init]] + list(np.mean(data[np.arange(i,len(signatures)*len(inits),len(inits)),:], axis=0)))
	

def heuristic_init(path, datasets, signatures, inits, metrics, n):
	metric = "l1"
	window = 5
	data = -1*np.ones((len(signatures)*len(inits), len(datasets)))
	for d, dataset in enumerate(datasets):
		for s, signature in enumerate(signatures):
			dmatrices = None
			for init in inits:
				name = "dtgw_%s_%s_%s_w%d.distances" % (signature, init, metric, window)
				matrix = np.loadtxt(os.path.join(path, dataset, name))
				# n = matrix.shape[0]
				dmatrix = matrix[:n,:n]
				if dmatrices is None:
					dmatrices = dmatrix[:,:,np.newaxis]
				else:
					dmatrices = np.concatenate((dmatrices, dmatrix[:,:,np.newaxis]), axis=2)
			if dmatrices is not None:
				dmin = np.min(dmatrices, axis=2)
				print(dmin)
				print(np.sum(dmin))
				for i, init in enumerate(inits):
					print(dmatrices[:,:,i])
					print(dmatrices[:,:,i] - dmin)
					print(np.sum(dmatrices[:,:,i] - dmin))
					rel_error = np.sum(dmatrices[:,:,i] - dmin)/np.sum(dmin)
					# rel_error = np.sum((dmatrices[:,:,j] - dmin)/dmin)
					data[s*len(inits)+i, d] = rel_error*100
		print(data)
	translate = {"optimistic_matching": "$\\sigma_{\\text{opt}}$",
	"sigma*": "$\\sigma^*$", "diagonal_warping": "swp", "optimistic_warping": "owp"}
	with open(os.path.join(path, "diff_heuristic_signature_init.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["signature", "init"] + datasets)
		for s, signature in enumerate(signatures):
			for i, init in enumerate(inits):
				writer.writerow([signature, translate[init]] + list(data[s*len(inits)+i,:]))
	with open(os.path.join(path, "diff_heuristic_init.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["init"] + datasets)
		for i, init in enumerate(inits):
			mean = np.mean(data[np.arange(i,len(signatures)*len(inits),len(inits)),:], axis=0)
			writer.writerow([translate[init]] + list(mean))



def wp_distances(path, datasets, signatures, windows):
	window, metric, n = 100, "l1", 50
	wps_max = np.zeros(len(datasets))
	wps_mean = np.zeros(len(datasets))
	wps_p80 = np.zeros(len(datasets))
	# wps_min = np.zeros(len(datasets))
	for d, dataset in enumerate(datasets):
		dataloader = DisseminationDataloader(dataset, "../datasets/dissemination")
		tgraphs, _ = dataloader.loadtxt(n)
		lifetimes = [tgraph.tadj.shape[0] for tgraph in tgraphs]
		lifetimes_diff = np.zeros((n, n))
		for i in range(n):
			for j in range(i+1,n):
				lifetimes_diff[i,j] = lifetimes_diff[j,i] = abs(lifetimes[i]-lifetimes[j])
		signatures = ["subtrees1"]
		for signature in signatures:
			name = "dtgw_%s_diagonal_warping_%s_w%d_n%d.wps" % (signature, metric, window, n)
			wps = np.loadtxt(os.path.join(path, dataset, name))
			dev = wps - lifetimes_diff
		wps_mean[d] = np.mean(dev)
		wps_max[d] = np.max(dev)
		wps_p80[d] = np.percentile(dev, 80)
		# wps_min[d] = np.min(dev)
	with open(os.path.join(path, "wps.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["desc"] + datasets)
		# writer.writerow(["$\\min$"] + list(wps_min))
		writer.writerow(["$\\varnothing$"] + list(wps_mean))
		writer.writerow(["$p80$"] + list(wps_p80))
		writer.writerow(["$\\max$"] + list(wps_max))


def relative_wp_distances(path, datasets, signatures):
	window, metric, n = 1, "l1", 20
	wps_max = np.zeros(len(datasets))
	wps_mean = np.zeros(len(datasets))
	wps_p80 = np.zeros(len(datasets))
	wps_p90 = np.zeros(len(datasets))
	# wps_min = np.zeros(len(datasets))
	for d, dataset in enumerate(datasets):
		dataloader = DisseminationDataloader(dataset, "../datasets/dissemination")
		tgraphs, _ = dataloader.loadtxt(n)
		lifetimes = [tgraph.tadj.shape[0] for tgraph in tgraphs]
		lifetimes_diff = np.zeros((n, n))
		lifetimes_max = np.ones((n, n))
		for i in range(n):
			for j in range(i+1,n):
				lifetimes_diff[i,j] = lifetimes_diff[j,i] = abs(lifetimes[i]-lifetimes[j])
				lifetimes_max[i,j] = lifetimes_max[j,i] = max(lifetimes[i],lifetimes[j])
		signatures = ["subtrees1"]
		for signature in signatures:
			name = "dtgw_%s_diagonal_warping_%s_w100_n%d.wps" % (signature, metric, n)
			wps = np.loadtxt(os.path.join(path, dataset, name))
			dev = (wps - lifetimes_diff)/lifetimes_max
		wps_mean[d] = np.mean(dev)
		wps_max[d] = np.max(dev)
		wps_p80[d] = np.percentile(dev, 80)
		wps_p90[d] = np.percentile(dev, 90)
		# wps_min[d] = np.min(dev)
	with open(os.path.join(path, "relative_wps.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["desc"] + datasets)
		# writer.writerow(["$\\min$"] + list(wps_min))
		writer.writerow(["$\\varnothing$"] + list(wps_mean))
		writer.writerow(["$p80$"] + list(wps_p80))
		writer.writerow(["$p90$"] + list(wps_p90))
		writer.writerow(["$\\max$"] + list(wps_max))


def window_solution_size(path, datasets, signatures, windows):
	metric, n = "l1", 50
	signature = "subtrees1"
	for d, dataset in enumerate(datasets):
		all_distances = np.full((n,n), np.inf)[:,:,np.newaxis]
		for window in windows:
			name = "dtgw_%s_diagonal_warping_%s_w%d_n%d.distances" % (signature, metric, window, n)
			distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
			all_distances = np.concatenate((all_distances, distances[:,:,np.newaxis]), axis=2)
		min_distances = np.min(all_distances, axis=2)
		with open(os.path.join(path, dataset, "window_solution_size.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["window", "error"])
			for window in windows:
				name = "dtgw_%s_diagonal_warping_%s_w%d_n%d.distances" % (signature, metric, window, n)
				distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
				rel_error = np.mean((distances - min_distances)/(distances + np.eye(n)))
				writer.writerow([window, rel_error])


def relative_window_solution_size(path, datasets, signatures, windows):
	metric, n = "l1", 20
	signature = "subtrees1"
	for d, dataset in enumerate(datasets):
		all_distances = np.full((n,n), np.inf)[:,:,np.newaxis]
		for window in windows:
			name = "dtgw_%s_diagonal_warping_%s_w%s_n%d.distances" % (signature, metric, ("%.2f" % window).replace('.', ''), n)
			distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
			all_distances = np.concatenate((all_distances, distances[:,:,np.newaxis]), axis=2)
		min_distances = np.min(all_distances, axis=2)
		with open(os.path.join(path, dataset, "relative_window_solution_size.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["window", "error"])
			for window in windows:
				name = "dtgw_%s_diagonal_warping_%s_w%s_n%d.distances" % (signature, metric, ("%.2f" % window).replace('.', ''), n)
				distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
				rel_error = np.mean((distances - min_distances)/(distances + np.eye(n)))
				writer.writerow([window, rel_error])


def window_runningtime(path, datasets, signatures, windows):
	metric, n = "l1", 50
	signature = "subtrees1"
	for d, dataset in enumerate(datasets):
		with open(os.path.join(path, dataset, "window_runningtime.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["window", "time"])
			for window in windows:
				name = "dtgw_%s_diagonal_warping_%s_w%d_n%d.time" % (signature, metric, window, n)
				file = open(os.path.join(path, dataset, name), "r")
				time = float(file.read()[:-2])/1000
				time /= (n*n-n)/2
				writer.writerow([window, time])


def relative_window_runningtime(path, datasets, signatures, windows):
	metric, n = "l1", 20
	signature = "subtrees1"
	for d, dataset in enumerate(datasets):
		dataloader = DisseminationDataloader(dataset, "../datasets/dissemination")
		tgraphs, _ = dataloader.loadtxt(n)
		lifetimes = [tgraph.tadj.shape[0] for tgraph in tgraphs]
		max_lifetimes = np.ones((n,n))
		for i in range(n):
			for j in range(i+1,n):
				max_lifetimes[i,j] = max_lifetimes[j,i] = max(lifetimes[i],lifetimes[j])
		with open(os.path.join(path, dataset, "relative_window_runningtime.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["window", "time"])
			for window in windows:
				name = "dtgw_%s_diagonal_warping_%s_w%s_n%d.time" % (signature, metric, ("%.2f" % window).replace('.', ''), n)
				file = open(os.path.join(path, dataset, name), "r")
				time = float(file.read()[:-2])/1000
				time /= (n*n-n)/2
				writer.writerow([window, time])


def running_times_iterations(path, datasets, iterations):
	metric, n = "l1", 20
	signature = "subtrees1"
	window = 0.1
	for d, dataset in enumerate(datasets):
		dataloader = DisseminationDataloader(dataset, "../datasets/dissemination")
		tgraphs, _ = dataloader.loadtxt(n)
		with open(os.path.join(path, dataset, "iterations_runningtime.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["iterations", "time"])
			for iteration in iterations:
				name = "dtgw_%s_diagonal_warping_%s_w%s_i%d_n%d.time" % (signature, metric, ("%.2f" % window).replace('.', ''), iteration, n)
				file = open(os.path.join(path, dataset, name), "r")
				time = float(file.read()[:-2])/1000
				time /= (n*n-n)/2
				writer.writerow([iteration, time])



def solution_size_iterations(path, datasets, iterations):
	metric, n = "l1", 20
	signature = "subtrees1"
	window = 0.1
	for d, dataset in enumerate(datasets):
		all_distances = np.full((n,n), np.inf)[:,:,np.newaxis]
		for iteration in iterations:
			name = "dtgw_%s_diagonal_warping_%s_w%s_i%d_n%d.distances" % (signature, metric, ("%.2f" % window).replace('.', ''), iteration, n)
			distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
			all_distances = np.concatenate((all_distances, distances[:,:,np.newaxis]), axis=2)
		min_distances = np.min(all_distances, axis=2)
		with open(os.path.join(path, dataset, "iterations_solution_size.csv"), "w") as file:
			writer = csv.writer(file, delimiter=';')
			writer.writerow(["iteration", "error"])
			for iteration in iterations:
				name = "dtgw_%s_diagonal_warping_%s_w%s_i%d_n%d.distances" % (signature, metric, ("%.2f" % window).replace('.', ''), iteration, n)
				distances = np.loadtxt(os.path.join(path, dataset, name))[:n,:n]
				rel_error = np.mean((distances - min_distances)/(distances + np.eye(n)))
				writer.writerow([iteration, rel_error])


def iterations(path, datasets):
	window, metric, n = 0.1, "l1", 20
	signature = "subtrees1"
	iterations_max = np.zeros(len(datasets))
	iterations_mean = np.zeros(len(datasets))
	iterations_p80 = np.zeros(len(datasets))
	iterations_p90 = np.zeros(len(datasets))
	iterations_min = np.zeros(len(datasets))
	for d, dataset in enumerate(datasets):
		name = "dtgw_%s_diagonal_warping_%s_w%s_n%d.iterations" % (signature, metric, ("%.2f" % window).replace('.', ''), n)
		iterations = np.loadtxt(os.path.join(path, dataset, name))
		iterations = iterations[np.triu_indices(n, k=1)].flatten()
		iterations_min[d] = np.min(iterations)
		iterations_mean[d] = np.mean(iterations)
		iterations_max[d] = np.max(iterations)
		iterations_p80[d] = np.percentile(iterations, 80)
		iterations_p90[d] = np.percentile(iterations, 90)
	with open(os.path.join(path, "iterations.csv"), "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["desc"] + datasets)
		writer.writerow(["$\\min$"] + list(iterations_min))
		writer.writerow(["$\\varnothing$"] + list(iterations_mean))
		# writer.writerow(["$p80$"] + list(iterations_p80))
		# writer.writerow(["$p90$"] + list(iterations_p90))
		writer.writerow(["$\\max$"] + list(iterations_max))


# def classification_results(path, datasets, classifiers, signatures, metrics):
# 	window, iterations, number = 0.2, 5, 100

# 	signatures = ["subtrees0", "subtrees1", "subtrees2"] + ["walks1", "walks2"]
# 	dtgw_algs = ["dtgw_%s_diagonal_warping_%s_w%s_i%d_n%d" % (signature, metric, ("%.2f" % window).replace('.', ''), iterations, number) for signature in signatures for metric in metrics]
# 	tkernel_algs = ["__%s_%d" % (alg, k) for alg in ["SEKS", "SEWL", "LGKS", "LGWL"] for k in [1,2,3]]
# 	algorithms = dtgw_algs + tkernel_algs

# 	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$",
# 				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL"}
# 	for task in ["_ct1", "_ct2"]:
# 		for classifier in classifiers:
# 			accuracies = -np.ones((len(algorithms), len(datasets)))
# 			variances = -np.ones((len(algorithms), len(datasets)))
# 			for i, dataset in enumerate(datasets):
# 				try:
# 					results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
# 				except Exception:
# 					continue
# 				# number = min(number, )
# 				if dataset == "mit":
# 					number = 97
# 				else:
# 					number = 100
# 				dtgw_algs = ["dtgw_%s_diagonal_warping_%s_w%s_i%d_n%d" % (signature, metric, ("%.2f" % window).replace('.', ''), iterations, number) for signature in signatures for metric in metrics]
# 				j = 0
# 				for algorithm in dtgw_algs:
# 					try:
# 						accuracies[j, i], variances[j, i] = results[algorithm]
# 						j += 1
# 					except Exception:
# 						j += 1
# 						continue
# 				for algorithm in tkernel_algs:
# 					try:
# 						accuracies[j, i], variances[j, i] = results[dataset + task + algorithm]
# 						j += 1
# 					except Exception:
# 						j += 1
# 						continue
			
# 			file = open(os.path.join(path, "accuracies_%s%s.csv" % (classifier, task)), "w")
# 			writer = csv.writer(file, delimiter=',')
# 			writer.writerow(["algorithm", "variant", "signature", "metric", "k"] + datasets)
# 			best_scores = np.argmax(accuracies, axis=0)
# 			i = 0
# 			for algorithm in dtgw_algs:
# 				params = algorithm.split('_')
# 				signature, k, metric = params[1][:-1], int(params[1][-1]), params[4]
# 				row = ["dtgw", "%s-%s" % (signature, translate[metric]), signature, metric, k]
# 				for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
# 					if best_scores[j] == i:
# 						row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
# 					else:
# 						row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
# 				writer.writerow(row)
# 				i += 1
# 			for algorithm in tkernel_algs:
# 				alg, k = algorithm[2:6], int(algorithm[7])
# 				row = ["tkernel", translate[alg], "", "", k]
# 				for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
# 					if best_scores[j] == i:
# 						row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
# 					else:
# 						row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
# 				writer.writerow(row)
# 				i += 1


def classification_results_dtgw(path, datasets, classifiers, signatures, metrics):
	window, iterations, number = 0.2, 5, 100

	signatures = ["subtrees0", "subtrees1", "subtrees2"] + ["walks1", "walks2"]
	dcosts = [0,1,2]
	dtgw_algs = ["dtgw_%s_c%d_diagonal_warping_%s_w%s_i%d_n%d" % (signature, c, metric, ("%.2f" % window).replace('.', ''), iterations, number) for signature in signatures for c in dcosts for metric in metrics]
	algorithms = dtgw_algs

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$",
				 # "subtrees": "$\\sigma$", "walks": "$\\omega$",
				 "subtrees": "subtree", "walks": "walk",
				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL",
				 "tkgw": "TGK-$\\land$", "tkg10": "TGK-$\\star$", "tkg11": "TGK-all"}

	for task in ["_ct1", "_ct2"]:
		accuracies = -np.ones((len(algorithms), len(datasets)))
		variances = -np.ones((len(algorithms), len(datasets)))
		for classifier in classifiers:
			for i, dataset in enumerate(datasets):
				try:
					results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
				except Exception:
					continue
				# number = min(number, )
				if dataset == "mit":
					dtgw_algs = ["dtgw_%s_c%d_diagonal_warping_%s_w%s_i%d_n97" % (signature, c, metric, ("%.2f" % window).replace('.', ''), iterations) for signature in signatures for c in [0,0,0] for metric in metrics]
				elif dataset == "infectious":
					dtgw_algs = ["dtgw_%s_c%d_diagonal_warping_%s_w%s_i%d_n%d" % (signature, c, metric, ("%.2f" % window).replace('.', ''), iterations, number) for signature in signatures for c in [0,0,0] for metric in metrics]
				else:
					dtgw_algs = ["dtgw_%s_c%d_diagonal_warping_%s_w%s_i%d_n%d" % (signature, c, metric, ("%.2f" % window).replace('.', ''), iterations, number) for signature in signatures for c in dcosts for metric in metrics]

				j = 0
				for algorithm in dtgw_algs:
					try:
						acc, var = results[algorithm]
						if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
							accuracies[j, i], variances[j, i] = acc, var
						j += 1
					except Exception:
						j += 1
						continue
			
		print(accuracies)

		c_text = ["0", "$f$", "$f_N$"]
		file = open(os.path.join(path, "dtgw_accuracies%s.csv" % (task)), "w")
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["algorithm", "variant", "signature", "metric", "k", "c"] + datasets)
		best_scores = np.argmax(accuracies, axis=0)
		i = 0
		for algorithm in dtgw_algs:
			params = algorithm.split('_')
			signature, k, c, metric = params[1][:-1], int(params[1][-1]), int(params[2][1]), params[5]
			# row = ["dtgw", "%s-%s" % (signature, translate[metric]), signature, translate[metric], k, c_text[c]]
			row = ["dtgw", translate[signature], signature, translate[metric], k, c_text[c]]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1


def classification_results_tkernels(path, datasets, classifiers, signatures, metrics):
	tkernel_algs = ["__%s_%d" % (alg, k) for alg in ["SEKS", "SEWL", "LGKS", "LGWL"] for k in [1,2,3]]
	tgraphlets = ["__tkg%s_%d" % (v, k) for v in ["10", "11", "w"] for k in [1,2,3]]
	algorithms = tkernel_algs + tgraphlets

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$",
				 "subtrees": "$\\sigma$", "walks": "$\\omega$",
				 "SEKS": "SE-RW", "SEWL": "SE-WL", "LGKS": "LG-RW", "LGWL": "LG-WL",
				 "tkgw": "TGK-$\\land$", "tkg10": "TGK-$\\star$", "tkg11": "TGK-all"}

	for task in ["_ct1", "_ct2"]:
		accuracies = -np.ones((len(algorithms), len(datasets)))
		variances = -np.ones((len(algorithms), len(datasets)))
		for classifier in classifiers:
			for i, dataset in enumerate(datasets):
				try:
					results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
				except Exception:
					continue
				j=0
				for algorithm in tkernel_algs:
					try:
						acc, var = results[dataset + task + algorithm]
						if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
							accuracies[j, i], variances[j, i] = acc, var
						j += 1
					except Exception:
						j += 1
						continue
				for algorithm in tgraphlets:
					try:
						acc, var = results[dataset + task + algorithm]
						if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
							accuracies[j, i], variances[j, i] = acc, var
						j += 1
					except Exception:
						j += 1
						continue

		file = open(os.path.join(path, "tkernels_accuracies%s.csv" % (task)), "w")
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["algorithm", "variant", "k"] + datasets)
		best_scores = np.argmax(accuracies, axis=0)
		i = 0
		for algorithm in tkernel_algs:
			alg, k = algorithm[2:6], int(algorithm[7])
			row = ["tkernel", translate[alg], k]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1
		for algorithm in tgraphlets:
			variant, k = algorithm[2:-2], int(algorithm[-1])
			row = ["tkernel", translate[variant], k]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1


def best_classification_results(path, datasets, classifiers, signatures, metrics):
	window, iterations, number = 0.2, 5, 100
	# tgalgs = ["SEKS", "SEWL", "LGKS", "LGWL"]
	tgalgs = ["SE", "LG"]
	talgs = ["tkg"]

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$",
				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL", "tkg": "TGK"}

	for task in ["_ct1", "_ct2"]:
		accuracies = -np.ones((len(signatures) + len(tgalgs) + len(talgs), len(datasets)))
		variances = -np.ones((len(signatures) + len(tgalgs) + len(talgs), len(datasets)))
		for d, dataset in enumerate(datasets):
			if dataset == "mit":
				if task == "_ct1":
					number = 95
				else:
					number = 89
			else:
				number = 100
			j = 0
			for signature in signatures:
				best_acc, best_var = -100, 100
				for classifier in classifiers:
					for metric in metrics:
						for dcost in [0,1,2]:
							for k in [0,1,2,3]:
								try:
									results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
									algorithm = "dtgw_%s%d_c%d_diagonal_warping_%s_w%s_i%d_n%d" % (signature, k, dcost, metric, ("%.2f" % window).replace('.', ''), iterations, number)
									acc, var = results[algorithm]
									if acc > best_acc or (acc == best_acc and var < best_var):
										best_acc, best_var = acc, var
								except Exception:
									continue
				accuracies[j, d], variances[j, d] = best_acc, best_var
				j += 1
			for alg in tgalgs:
				best_acc, best_var = -100, 100
				for classifier in classifiers:
					for graph_kernel in ["KS", "WL"]:
						for k in [1,2,3]:
							try:
								results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
								algorithm = "%s__%s%s_%d" % (task, alg, graph_kernel, k)
								acc, var = results[dataset + algorithm]
								if acc > best_acc or (acc == best_acc and var < best_var):
									best_acc, best_var = acc, var
							except Exception:
								continue
				accuracies[j, d], variances[j, d] = best_acc, best_var
				j += 1
			for alg in talgs:
				best_acc, best_var = -100, 100
				for classifier in classifiers:
					for variant in ["10", "11", "w"]:
						for k in [0,1,2,3]:
							try:
								results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
								algorithm = "%s__%s%s_%d" % (task, alg, variant, k)
								acc, var = results[dataset + algorithm]
								if acc > best_acc or (acc == best_acc and var < best_var):
									best_acc, best_var = acc, var
							except Exception:
								continue
				accuracies[j, d], variances[j, d] = best_acc, best_var
				j += 1
		print(accuracies)

		file = open(os.path.join(path, "accuracies2%s.csv" % task), "w")
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["algorithm", "variant"] + datasets)

		best_scores = np.argmax(accuracies, axis=0)
		i = 0
		for signature in signatures:
			row = ["dtgw", signature]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1
		for algorithm in tgalgs:
			# row = ["tkernel", translate[algorithm]]
			row = ["tkernel", algorithm]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1
		for algorithm in talgs:
			row = ["tkernel", translate[algorithm]]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1


def best_running_times(path, datasets, classifiers, signatures, metrics):
	window, iterations, number = 0.2, 5, 100
	tgalgs = ["SE", "LG"]
	talgs = ["tkg"]

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$",
				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL", "tkg": "TGK"}

	for task in ["_ct1", "_ct2"]:
		running_times = np.empty((len(signatures) + len(tgalgs) + len(talgs), len(datasets)))
		for d, dataset in enumerate(datasets):
			if dataset == "mit":
				if task == "_ct1":
					number = 95
				else:
					number = 89
			else:
				number = 100
			j = 0
			for signature in signatures:
				best_acc, best_var, time = -100, 100, np.inf
				for classifier in classifiers:
					for metric in metrics:
						for dcost in [0,1,2]:
							for k in [0,1,2]:
								try:
									results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
									algorithm = "dtgw_%s%d_c%d_diagonal_warping_%s_w%s_i%d_n%d" % (signature, k, dcost, metric, ("%.2f" % window).replace('.', ''), iterations, number)
									acc, var = results[algorithm]
									if acc > best_acc or (acc == best_acc and var < best_var):
										best_acc, best_var = acc, var
										with open(os.path.join(path, dataset + task, algorithm + ".time"), "r") as file:
											time = float(file.read()[:-2])
								except Exception:
									continue
				running_times[j, d] = time/1000
				j += 1
			for alg in tgalgs:
				best_acc, best_var, time = -100, 100, np.inf
				for classifier in classifiers:
					for graph_kernel in ["KS", "WL"]:
						for k in [1,2,3]:
							try:
								results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
								algorithm = "%s__%s%s_%d" % (task, alg, graph_kernel, k)
								acc, var = results[dataset + algorithm]
								if acc > best_acc or (acc == best_acc and var < best_var):
									best_acc, best_var = acc, var
									with open(os.path.join(path, dataset + task, dataset + algorithm + ".gram.time"), "r") as file:
										time = float(file.readline().split(' ')[0])
							except Exception:
								continue
				running_times[j, d] = time/1000
				j += 1
			for alg in talgs:
				best_acc, best_var, time = -100, 100, np.inf
				for classifier in classifiers:
					for variant in ["10", "11", "w"]:
						for k in [1,2,3]:
							try:
								results = json.load(open(os.path.join(path, dataset + task, "accuracies_%s.json" % classifier), "r"))
								algorithm = "%s__%s%s_%d" % (task, alg, variant, k)
								acc, var = results[dataset + algorithm]
								if acc > best_acc or (acc == best_acc and var < best_var):
									best_acc, best_var = acc, var
									with open(os.path.join(path, dataset + task, dataset + algorithm + ".gram.time"), "r") as file:
										time = float(file.readline().split(' ')[0])
							except Exception:
								continue
				running_times[j, d] = time/1000
				j += 1
		
		print("rtimes")
		print(running_times)
		file = open(os.path.join(path, "running_times2%s.csv" % task), "w")
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["algorithm", "variant"] + datasets)

		i = 0
		for signature in signatures:
			row = ["dtgw", signature] + list(running_times[i,:])
			writer.writerow(row)
			i += 1
		for algorithm in tgalgs:
			row = ["tkernel", algorithm] + list(running_times[i,:])
			writer.writerow(row)
			i += 1
		for algorithm in talgs:
			row = ["tkernel", translate[algorithm]] + list(running_times[i,:])
			writer.writerow(row)
			i += 1


if __name__ == "__main__":
	path = "../output/dissemination"
	# running_times(path)
	# statistics_datasets(path)
	# datasets = ["infectious_ct1", "mit_ct1", "tumblr_ct1", "highschool_ct1", "facebook_ct1", "dblp_ct1"]
	# datasets = ["infectious_ct2", "mit_ct2", "tumblr_ct2", "highschool_ct2", "facebook_ct2", "dblp_ct2"]
	# datasets = ["infectious_ct1", "tumblr_ct1", "highschool_ct1", "facebook_ct1"]
	# datasets = ["infectious_ct1", "mit_ct1"]
	# signatures = ["subtrees%d" % i for i in range(0,3)] + ["walks%d" % i for i in range(1,3)]
	# signatures = ["subtrees%d" % i for i in range(0,2)] + ["walks%d" % i for i in range(1,2)]
	signatures = ["subtrees%d" % i for i in range(0,2)] + ["walks2"]
	# inits = ["diagonal_warping", "optimistic_warping", "sigma*", "optimistic_matching"]
	inits = ["diagonal_warping"]
	metrics = ["l1", "l2"]
	n = 100

	# heuristic_init(path, datasets, signatures, inits, metrics, n)
	# running_times_init(path, datasets, signatures, inits, metrics)
	# windows = [1, 2, 3, 5, 10, 20, 50, 100]
	# relative_wp_distances(path, datasets, signatures)
	# windows = [2, 3, 5, 10, 20, 50, 100]
	# windows=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
	# window_solution_size(path, datasets, signatures, windows)
	# window_runningtime(path, datasets, signatures, windows)
	# relative_window_solution_size(path, datasets, signatures, windows)
	# relative_window_runningtime(path, datasets, signatures, windows)
	
	# iterations(path, datasets)
	# iterations = [2,3,4,5,7,10,20,50]
	# running_times_iterations(path, datasets, iterations)
	# solution_size_iterations(path, datasets, iterations)
	datasets = ["mit", "infectious", "tumblr", "highschool", "dblp", "facebook"]
	classifiers = ["kNN", "SVM_linear", "SVM_rbf"]
	# classifiers = ["kNN"]
	signatures = ["subtrees", "walks"]
	# signatures = ["subtrees2"]
	# classification_results(path, datasets, classifiers, signatures, metrics)
	# classification_results_dtgw(path, datasets, classifiers, signatures, metrics)
	classification_results_tkernels(path, datasets, classifiers, signatures, metrics)
	# best_classification_results(path, datasets, classifiers, signatures, metrics)
	# best_running_times(path, datasets, classifiers, signatures, metrics)





