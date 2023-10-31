import os
import csv
import json
import numpy as np

# import brains.brain_dataloader as braindataloader


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


# def statistics_datasets(dataset_path, datasets, n, roi, output_path):
# 	data = np.empty((7, len(datasets)))
# 	for j, dataset in enumerate(datasets):
# 		if dataset == "development":
# 			dataloader = braindataloader.BrainDevelopementDataloader("development", dataset_path, n, roi)
# 		if dataset == "abide":
# 			dataloader = braindataloader.AbideDataloader("abide", dataset_path, n, roi)
# 		if dataset == "adhd":
# 			dataloader = braindataloader.ADHDDataloader("adhd", dataset_path, n, roi)

# 		tadjs, labels = dataloader.get_data()
# 		n = len(tadjs)

# 		v, e, t = np.empty(n), np.empty(n), np.empty(n)
# 		for i, tadj in enumerate(tadjs):
# 			t[i] = tadj.shape[0]
# 			v[i] = tadj.shape[1]
# 			e[i] = np.sum(tadj)/2
# 		data[0,j] = n
# 		data[1,j] = v.max()
# 		data[2,j] = v.mean()
# 		data[3,j] = e.max()
# 		data[4,j] = e.mean()
# 		data[5,j] = t.max()
# 		data[6,j] = t.mean()

# 	with open(os.path.join(output_path, "statistics.csv"), "w") as file:
# 		writer = csv.writer(file, delimiter=',')
# 		writer.writerow(["properties"] + datasets)
# 		writer.writerow(["$|\\mathcal{D}|$"] + list(data[0,:]))
# 		# writer.writerow(["$\\max |V|$"] + list(data[1,:]))
# 		# writer.writerow(["$\\varnothing |V|$"] + list(data[2,:]))
# 		# writer.writerow(["$\\max |\\mathcal{E}|$"] + ["%1.1f" % data[3,i] for i in range(len(datasets))])
# 		# writer.writerow(["$\\varnothing |\\mathcal{E}|$"] + ["%1.1f" % data[4,i] for i in range(len(datasets))])
# 		writer.writerow(["$\\max |\\mathcal{E}|$"] + list(data[3,:]))
# 		writer.writerow(["$\\varnothing |\\mathcal{E}|$"] + list(data[4,:]))
# 		writer.writerow(["$\\max T$"] + list(data[5,:]))
# 		writer.writerow(["$\\varnothing T$"] + list(data[6,:]))



def tgw_classification_results(path, datasets, classifiers, signatures, metrics):
	window = 0.2
	translate = {"l1": "$L_1$", "l2": "$L_2$"}

	accuracies = -np.ones((len(signatures)*len(metrics)*3, len(datasets)))
	variances = -np.ones((len(signatures)*len(metrics)*3, len(datasets)))
	for classifier in classifiers:
		for i, dataset in enumerate(datasets):
			try:
				results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
			except Exception:
				continue
			j = 0
			for signature in signatures:
				for k in [1,2,3]:
					for metric in metrics:
						alg = "tgw_%s%d_%s_w%s" % (signature, k, metric, ("%.2f" % window).replace('.', ''))
						acc, var = results[alg]
						if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
							accuracies[j, i], variances[j, i] = acc, var
						j +=1
			
	print(accuracies)

	file = open(os.path.join(path, "tgw_accuracies.csv"), 'w')
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["signature", "metric", "k"] + datasets)
	best_scores = np.argmax(accuracies, axis=0)
	i = 0
	for signature in signatures:
		for k in [1,2,3]:
			for metric in metrics:
				row = [signature, translate[metric], k]
				for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
					if best_scores[j] == i:
						row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
					else:
						row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
				writer.writerow(row)
				i += 1


def tkernel_classification_results(path, datasets, classifiers, signatures, metrics):
	window = 0.2
	translate = {"l1": "$L_1$", "l2": "$L_2$"}

	accuracies = -np.ones((2*1+4*3, len(datasets)))
	variances = -np.ones((2*1+4*3, len(datasets)))
	for classifier in classifiers:
		for i, dataset in enumerate(datasets):
			try:
				results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
			except Exception:
				continue
			j = 0
			# tw
			# for metric in metrics:
			# 	alg = "tw_%s" % (metric)
			# 	acc, var = results[alg]
			# 	if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
			# 		accuracies[j, i], variances[j, i] = acc, var
			# 	j += 1
			# corr
			# acc, var = results["corr"]
			# if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
			# 	accuracies[j, i], variances[j, i] = acc, var
			# j += 1
			# tkernel
			for kernel in ["SE"]:
				for k in [1,2]:
					alg = "%s__%sKS_%d" % (dataset, kernel, k)
					acc, var = results[alg]
					if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
						accuracies[j, i], variances[j, i] = acc, var
					j +=1
			for k in [2,3,5,10]:
				for S in [100, 1000, 10000]:
					alg = "%s__TMAP%d_%d" % (dataset, S, k)
					acc, var = results[alg]
					if acc > accuracies[j, i] or (acc == accuracies[j, i] and var < variances[j, i]):
						accuracies[j, i], variances[j, i] = acc, var
					j +=1
			
	print(accuracies)

	file = open(os.path.join(path, "tkernel_accuracies.csv"), 'w')
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["kernel", "k", "S"] + datasets)
	best_scores = np.argmax(accuracies, axis=0)
	i = 0
	# for metric in metrics:
	# 	row = ["tw", "", "", metric, "", ""]
	# 	for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
	# 		if best_scores[j] == i:
	# 			row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
	# 		else:
	# 			row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
	# 	writer.writerow(row)
	# 	i += 1
	# row = ["corr", "", "", "", "", ""]
	# for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
	# 	if best_scores[j] == i:
	# 		row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
	# 	else:
	# 		row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
	# 	writer.writerow(row)
	for k in [1,2]:
		row = ["SE-RW", k, ""]
		for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
			if best_scores[j] == i:
				row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
			else:
				row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
		writer.writerow(row)
		i += 1
	for k in [2,3,5,10]:
		for S in [100, 1000, 10000]:
			row = ["Approx", k, S]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)
			i += 1	



def best_classification_results(path, datasets, classifiers, signatures, metrics):
	window, iterations, number = 0.2, 5, 100
	tgalgs = ["SE"]
	talgs = ["TMAP"]

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$", "SE": "SE-RW",
				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL", "tkg": "TGK",
				 "interaction": "interaction", "degree": "degree", "TMAP": "Approx."}

	accuracies = -np.ones((len(signatures) + 4, len(datasets)))
	variances = -np.ones((len(signatures) + 4, len(datasets)))
	for d, dataset in enumerate(datasets):
		j = 0
		best_acc, best_var = -100, 100
		for classifier in classifiers:
			for metric in metrics:
				try:
					results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
					algorithm = "tw_%s" % metric
					acc, var = results[algorithm]
					if acc > best_acc or (acc == best_acc and var < best_var):
						best_acc, best_var = acc, var
				except Exception:
					continue
		accuracies[j, d], variances[j, d] = best_acc, best_var
		j += 1	
		best_acc, best_var = -100, 100
		for classifier in classifiers:
			try:
				results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
				algorithm = "corr"
				acc, var = results[algorithm]
				if acc > best_acc or (acc == best_acc and var < best_var):
					best_acc, best_var = acc, var
			except Exception:
				continue
		accuracies[j, d], variances[j, d] = best_acc, best_var
		j += 1	
		for signature in signatures:
			best_acc, best_var = -100, 100
			for classifier in classifiers:
				for metric in metrics:
					for k in [1,2,3]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "tgw_%s%d_%s_w%s" % (signature, k, metric, ("%.2f" % window).replace('.', ''))
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
				for graph_kernel in ["KS"]:
					for k in [1,2,3]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "%s__%s%s_%d" % (dataset, alg, graph_kernel, k)
							acc, var = results[algorithm]
							if acc > best_acc or (acc == best_acc and var < best_var):
								best_acc, best_var = acc, var
						except Exception:
							continue
			accuracies[j, d], variances[j, d] = best_acc, best_var
			j += 1
		for alg in talgs:
			best_acc, best_var = -100, 100
			for classifier in classifiers:
				for S in [100, 1000, 10000]:
					for k in [1,2,3,4,5,6,7,8,9,10]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "%s__%s%d_%d" % (dataset, alg, S, k)
							acc, var = results[algorithm]
							if acc > best_acc or (acc == best_acc and var < best_var):
								best_acc, best_var = acc, var
						except Exception:
							continue
			accuracies[j, d], variances[j, d] = best_acc, best_var
			j += 1
	print(accuracies)

	file = open(os.path.join(path, "accuracies.csv"), "w")
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["algorithm", "variant"] + datasets)

	best_scores = np.argmax(accuracies, axis=0)
	i = 0
	row = ["tw", ""]
	for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
		if best_scores[j] == i:
			row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
		else:
			row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
	writer.writerow(row)
	i += 1
	row = ["corr", ""]
	for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
		if best_scores[j] == i:
			row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
		else:
			row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
	writer.writerow(row)
	i += 1
	for signature in signatures:
		row = ["tgw", signature]
		for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
			if best_scores[j] == i:
				row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
			else:
				row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
		writer.writerow(row)
		i += 1
	for algorithm in tgalgs:
		row = ["tkernel", translate[algorithm]]
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
	tgalgs = ["SE"]
	talgs = ["TMAP"]

	translate = {"l1": "$L_1$", "l2": "$L_2$", "dot": "$\\langle,\\rangle$", "SE": "SE",
				 "SEKS": "SE-KS", "SEWL": "SE-WL", "LGKS": "LG-KS", "LGWL": "DL-WL", "tkg": "TGK",
				 "interaction": "interaction", "degree": "degree", "TMAP": "Approx."}

	running_times = np.empty((len(signatures) + 4, len(datasets)))
	for d, dataset in enumerate(datasets):
		j = 0
		time = np.inf
		best_acc, best_var = -100, 100
		for classifier in classifiers:
			for metric in metrics:
				try:
					results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
					algorithm = "tw_%s" % metric
					acc, var = results[algorithm]
					if (acc > best_acc) or (acc == best_acc and var < best_var):
						with open(os.path.join(path, dataset, algorithm + ".time"), "r") as file:
							time = float(file.read()[:-2])
						best_acc, best_var = acc, var
				except Exception:
					continue
		running_times[j, d] = time/1000
		j += 1
		time = np.inf
		best_acc, best_var = -100, 100
		for classifier in classifiers:
			try:
				results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
				algorithm = "corr"
				acc, var = results[algorithm]
				if (acc > best_acc) or (acc == best_acc and var < best_var):
					best_acc, best_var = acc, var
					with open(os.path.join(path, dataset, algorithm + ".time"), "r") as file:
						time = float(file.read()[:-2])
			except Exception:
				continue
		running_times[j, d] = time/1000
		j += 1
		time = np.inf
		best_acc, best_var = -100, 100
		for signature in signatures:
			best_acc, best_var = -100, 100
			for classifier in classifiers:
				for metric in metrics:
					for k in [1,2,3]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "tgw_%s%d_%s_w%s" % (signature, k, metric, ("%.2f" % window).replace('.', ''))
							acc, var = results[algorithm]
							if (acc > best_acc) or (acc == best_acc and var < best_var):
								best_acc, best_var = acc, var
								with open(os.path.join(path, dataset, algorithm + ".time"), "r") as file:
									time = float(file.read()[:-2])
						except Exception:
							continue
			running_times[j, d] = time/1000
			j += 1
		time = np.inf
		best_acc, best_var = -100, 100
		for alg in tgalgs:
			best_acc, best_var = -100, 100
			for classifier in classifiers:
				for graph_kernel in ["KS"]:
					for k in [1,2,3]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "%s__%s%s_%d" % (dataset, alg, graph_kernel, k)
							acc, var = results[algorithm]
							if (acc > best_acc) or (acc == best_acc and var < best_var):
								best_acc, best_var = acc, var
								with open(os.path.join(path, dataset, algorithm + ".gram.time"), "r") as file:
									time = float(file.readline().split(' ')[0])
						except Exception:
							continue
			running_times[j, d] = time/1000
			j += 1
		time = np.inf
		best_acc, best_var = -100, 100
		for alg in talgs:
			best_acc, best_var = -100, 100
			for classifier in classifiers:
				for S in [100, 250, 500, 1000]:
					for k in [1,2,3]:
						try:
							results = json.load(open(os.path.join(path, dataset, "accuracies_%s.json" % classifier), "r"))
							algorithm = "%s__%s%d_%d" % (dataset, alg, S, k)
							acc, var = results[algorithm]
							if (acc > best_acc) or (acc == best_acc and var < best_var):
								best_acc, best_var = acc, var
								with open(os.path.join(path, dataset, algorithm + ".gram.time"), "r") as file:
									time = float(file.readline().split(' ')[0])
						except Exception:
							continue
			running_times[j, d] = time/1000
			j += 1
	print(running_times)

	file = open(os.path.join(path, "running_times.csv"), "w")
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["algorithm", "variant"] + datasets)

	# best_scores = np.argmax(running_times, axis=0)

	file = open(os.path.join(path, "running_times.csv"), "w")
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["algorithm", "variant"] + datasets)

	i = 0
	row = ["tw", ""] + list(running_times[i,:])
	writer.writerow(row)
	i += 1
	row = ["corr", ""] + list(running_times[i,:])
	writer.writerow(row)
	i += 1
	for signature in signatures:
		row = ["tgw", translate[signature]] + list(running_times[i,:])
		writer.writerow(row)
		i += 1
	for algorithm in tgalgs:
		row = ["tkernel", translate[algorithm]] + list(running_times[i,:])
		writer.writerow(row)
		i += 1
	for algorithm in talgs:
		row = ["tkernel", translate[algorithm]] + list(running_times[i,:])
		writer.writerow(row)
		i += 1


def dtgw_matchings(output_path, datasets, signatures, metrics):
	window = 0.2
	best_matchings_avg = np.zeros((len(signatures), len(datasets)))
	best_matchings_max = np.zeros((len(signatures), len(datasets)))
	for d, dataset in enumerate(datasets):
		for s, signature in enumerate(signatures):
			best_matching_avg, best_matching_max = -float('inf'), -float('inf')
			for metric in metrics:
				for k in [1,2,3]:
					name = "dtgw_%s%d_%s_w020" % (signature, k, metric)
					matching = np.loadtxt(os.path.join(output_path, dataset, name + ".matchings"))
					matrix = np.loadtxt(os.path.join(output_path, dataset, name + ".distances"))
					labels = matrix[:,matrix.shape[0]]
					indices = np.where(labels)
					matching = matching[indices]
					if np.mean(matching) > best_matching_avg:
						best_matching_avg, best_matching_max = np.mean(matching), np.max(matching)
			best_matchings_avg[s, d] = best_matching_avg
			best_matchings_max[s, d] = best_matching_max
	print(best_matchings_avg)
	print(best_matchings_max)

	file = open(os.path.join(output_path, "matchings.csv"), "w")
	writer = csv.writer(file, delimiter=',')
	writer.writerow(["signature", "desc"] + datasets)
	for s, signature in enumerate(signatures):
		row = [signature, "$\\varnothing$"] + list(best_matchings_avg[s,:]*100)
		writer.writerow(row)
		row = [signature, "$\\max$"] + list(best_matchings_max[s,:]*100)
		writer.writerow(row)





if __name__ == "__main__":
	dataset_path = os.path.join("..", "datasets", "brains")
	output_path = os.path.join("..", "output", "brains")

	datasets = ["abide", "development", "adhd"]
	roi = "atlas"
	n = 100

	# statistics_datasets(dataset_path, datasets, n, roi, output_path)

	signatures = ["interaction", "degree"]
	# inits = ["diagonal_warping"]
	metrics = ["l1", "l2"]
	

	# classifiers = ["kNN", "SVM_linear", "SVM_rbf"]
	classifiers = ["kNN", "SVM_linear"]
	# signatures = ["subtrees", "walks"]
	# tgw_classification_results(output_path, datasets, classifiers, signatures, metrics)
	# tkernel_classification_results(output_path, datasets, classifiers, signatures, metrics)
	# best_classification_results(output_path, datasets, classifiers, signatures, metrics)
	# best_running_times(output_path, datasets, classifiers, signatures, metrics)

	dtgw_matchings(output_path, datasets, signatures, metrics)




