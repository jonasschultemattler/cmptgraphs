import sys
import os
import csv
import json
import argparse
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

from train import train

# def load_data(file):
# 	with open(file) as fp:
# 		lines = fp.readlines()
# 		n = len(lines)
# 		gram_matrix = np.zeros((n, n))
# 		labels = np.zeros(n)
# 		for i, line in enumerate(lines):
# 			line_split = line.split(':')
# 			labels[i] = int(line_split[0].split(' ')[0])
# 			for j, column in enumerate(line_split[2:]):
# 				gram_matrix[i][j] = float(column.split(' ')[0])
# 	return gram_matrix, labels


# def load(file):
# 	m = np.loadtxt(file)
# 	n = m.shape[0]
# 	matrix, y = m[:,:n], m[:,n]
# 	if np.isnan(matrix).any() or not np.isfinite(matrix).all():
# 		raise ValueError("inf or nan in %s", file)
# 	return matrix, y



# def train_svms(output_path, dataset, distances, metrics, kernels, tgkernels_path, tgkernels):
# 	results = {}
# 	for kernel in kernels:
# 		print("training " + kernel + "...")
# 		K, y = load(os.path.join(output_path, dataset, kernel + ".gram"))
# 		acc, std = train_kernel_svm(K, y)
# 		results.update({kernel: [acc, std]})
# 	for distance in distances:
# 		print("training " + distance + "...")
# 		# config = distance
# 		for metric in metrics:
# 			config = "%s_%s_w10" % (distance, metric)
# 			D, y = load(os.path.join(output_path, dataset, config + ".distances"))
# 			acc, std = train_svm_distances(D, y)
# 			results.update({config: [acc, std]})
# 	for tgkernel in tgkernels:
# 		print("training " + tgkernel + "...")
# 		acc, std = train_svm_tgkernel(os.path.join(tgkernels_path, dataset + "__" + tgkernel))
# 		results.update({tgkernel: [acc, std]})
# 	if len(distances) > 0 or len(kernels) > 0:
# 		results.update({"dice": max(np.sum(y), len(y)-np.sum(y))})
# 	return results


# def train_svm(output_path, dataset, algorithm, signature, metric, window, k):
# 	print("training " + algorithm + "...")
# 	if algorithm in ["corr", "tadj"]:
# 		K, y = load(os.path.join(output_path, dataset, algorithm + ".gram"))
# 		acc, std = train_kernel_svm(K, y)
# 	elif algorithm in ["tgw"]:
# 		config = "%s_%s_%s_w%d" % (algorithm, signature, metric, window)
# 		if metric == "dot":
# 			K, y = load(os.path.join(output_path, dataset, config + ".gram"))
# 			acc, std = train_kernel_svm(K, y)
# 		else:
# 			D, y = load(os.path.join(output_path, dataset, config + ".distances"))
# 			acc, std = train_svm_distances(D, y)
# 	elif algorithm in ["SEKS"]:
# 		tgkernels_path = "tgkernel/release/"
# 		acc, std = train_svm_tgkernel(os.path.join(tgkernels_path, dataset + "__" + algorithm))
# 	return acc, std


# def rbf_distance_kernel(dmatrix, gamma=1):
# 	return np.exp(-gamma*dmatrix**2)


# def train_svm_distances(D, y, n=10, k=10, l=10):
# 	D /= np.max(D)

# 	accuracies, variances = np.zeros(n), np.zeros(n)
# 	for iteration in range(n):
# 		print("Iteration %d:" % iteration)
# 		Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]
# 		gammas = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10]

# 		training_accuracies, test_accuracies = np.zeros(k), np.zeros(k)

# 		outer_cv = KFold(n_splits=k, shuffle=True, random_state=iteration).split(D)
# 		for i, (train_index, test_index) in enumerate(outer_cv):
# 			D_train, y_train = D[train_index][:,train_index], y[train_index]
# 			D_test, y_test = D[test_index][:,train_index], y[test_index]

# 			inner_cv = KFold(n_splits=l, shuffle=True, random_state=i).split(D_train)
# 			for train_index_i, test_index_i in inner_cv:
# 				D_train_i, y_train_i = D_train[train_index_i][:,train_index_i], y_train[train_index_i]
# 				D_test_i, y_test_i = D_train[test_index_i][:,train_index_i], y_train[test_index_i]

# 				best_score, best_C, best_gamma = -1, -1, -1
# 				for c in Cs:
# 					svc = svm.SVC(kernel='precomputed', C=c)
# 					for gamma in gammas:
# 						K_train_i = rbf_distance_kernel(D_train_i, gamma=gamma)
# 						K_test_i = rbf_distance_kernel(D_test_i, gamma=gamma)
# 						svc.fit(K_train_i, y_train_i)
# 						score = np.mean(svc.score(K_test_i, y_test_i))
# 						if score > best_score:
# 							best_score, best_C, best_gamma = score, c, gamma

# 			K_train, K_test = rbf_distance_kernel(D_train, gamma=best_gamma), rbf_distance_kernel(D_test, gamma=best_gamma)
# 			svc = svm.SVC(kernel='precomputed', C=best_C)
# 			svc.fit(K_train, y_train)
# 			test_accuracies[i] = np.mean(svc.score(K_test, y_test))
# 			training_accuracies[i] = np.mean(svc.score(K_train, y_train))

# 			print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s, gamma %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C, best_gamma))

# 		print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
# 		print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

# 		accuracies[iteration] = test_accuracies.mean()
# 		variances[iteration] = test_accuracies.std()

# 	accuracy, variance = accuracies.mean()*100, variances.mean()*100
# 	print("Overall accuracy: %.1f+-%.1f" % (accuracy, variance))

# 	return accuracy, variance


# def train_svm_tgkernel(path, k=10, l=10, n=10):
# 	# hs = [0,1,2,3,4,5]
# 	hs = [0,1,2]
# 	best_h, best_score, best_variance = -1, -1, -1
	
# 	for h in hs:
# 		print("train svm for %s" % (path + "_" + str(h) + ".gram ..."))
# 		K, y = load_data(path + "_" + str(h) + ".gram")
# 		# K = K[:100,:100]
# 		# y = y[:100]
# 		accuracy, variance = train_kernel_svm(K, y, n, k, l)

# 		if accuracy > best_score:
# 			best_h, best_score, best_variance = h, accuracy, variance

# 	return best_score, best_variance


# def train_kernel_svm(K, y, n=10, k=10, l=10):
# 	Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]

# 	accuracies, variances = np.zeros(n), np.zeros(n)
# 	for iteration in range(n):
# 		print("Iteration %d:" % iteration)

# 		training_accuracies, test_accuracies = np.zeros(k), np.zeros(k)

# 		outer_cv = KFold(n_splits=k, shuffle=True, random_state=iteration).split(K)
# 		for i, (train_index, test_index) in enumerate(outer_cv):
# 			K_train, y_train = K[train_index][:,train_index], y[train_index]
# 			K_test, y_test = K[test_index][:,train_index], y[test_index]

# 			inner_cv = KFold(n_splits=l, shuffle=True, random_state=i).split(K_train)
# 			for train_index_i, test_index_i in inner_cv:
# 				best_c_score, best_C = -1, -1
# 				for c in Cs:
# 					K_train_i, y_train_i = K_train[train_index_i][:,train_index_i], y_train[train_index_i]
# 					K_test_i, y_test_i = K_train[test_index_i][:,train_index_i], y_train[test_index_i]

# 					svc = svm.SVC(kernel='precomputed', C=c)
# 					svc.fit(K_train_i, y_train_i)
# 					score = np.mean(svc.score(K_test_i, y_test_i))
# 					if score > best_c_score:
# 						best_c_score, best_C = score, c
			
# 			svc = svm.SVC(kernel='precomputed', C=best_C)
# 			svc.fit(K_train, y_train)
# 			test_accuracies[i] = np.mean(svc.score(K_test, y_test))
# 			training_accuracies[i] = np.mean(svc.score(K_train, y_train))

# 			# print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C))

# 		print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
# 		print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

# 		accuracies[iteration] = test_accuracies.mean()
# 		variances[iteration] = test_accuracies.std()

# 	accuracy, variance = accuracies.mean()*100, variances.mean()*100
# 	print("Overall accuracy: %.1f+-%.1f" % (accuracy, variance))

# 	return accuracy, variance


def save_result(result, output_path, dataset, classifier):
	res_file_path = os.path.join(output_path, dataset, "accuracies_%s.json" % classifier)
	if os.path.exists(res_file_path):
		with open(res_file_path, "r") as file:
			results = json.load(file)
		results.update(result)
		with open(res_file_path, "w") as file:
			json.dump(results, file)
	else:
		with open(res_file_path, "w+") as file:
			json.dump(result, file)



# def save_results(path, datasets, algorithms, translate):
# 	accuracies = np.empty((len(algorithms), len(datasets)))
# 	variances = np.empty((len(algorithms), len(datasets)))
# 	dataset_classes = {}
# 	for i, dataset in enumerate(datasets):
# 		results = json.load(open(os.path.join(path, dataset, "accuracies.json"), "r"))
# 		for j, algorithm in enumerate(algorithms):
# 			accuracies[j, i], variances[j, i] = results[algorithm]
# 		dataset_classes.update({dataset: results["dice"]})

# 	file = open(os.path.join(path, "accuracies.csv"), "w")
# 	writer = csv.writer(file, delimiter=',')
# 	# writer.writerow(["algorithm"] + ["%s [%.1f]" % (d, dataset_classes[d]) for d in datasets])
# 	writer.writerow(["algorithm"] + datasets)
# 	writer.writerow(["dice"] + [dataset_classes[dataset] for dataset in datasets])
# 	best_scores = np.argmax(accuracies, axis=0)
# 	for i, algorithm in enumerate(algorithms):
# 		row = [algorithm]
# 		for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
# 			if best_scores[j] == i:
# 				row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
# 			else:
# 				row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
# 		writer.writerow(row)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset")
	parser.add_argument("--classifier", choices=("SVM_linear", "SVM_rbf", "kNN"), default="SVM_linear", help="classifier")
	parser.add_argument("--alg", choices=("tw", "corr", "dtgw", "tgw", "tgembed", "SEKS", "TMAP"), default="tgw", help="algorithm to compute distances")
	parser.add_argument("--signature", choices=("degree", "interaction"), default="degree", help="vertex signature")
	parser.add_argument("--metric", choices=("l1", "l2", "dot"), default="l1", help="metric norm")
	parser.add_argument("--window", type=float, default=0.2, help="relative window size for time warping")
	parser.add_argument("--k", type=int, default=2, help="k step random walk kernel")
	parser.add_argument("--S", type=int, default=1000, help="number walks for approx walk kernel")
	args = parser.parse_args()

	output_path = "../output/brains"

	if args.alg == "tw":
		name = "tw_%s" % args.metric
		file = name + ".distances"
	if args.alg == "corr":
		name = "corr"
		file = name + ".gram"
	if args.alg in ["tgw"]:
		name = "%s_%s%d_%s_w%s" % (args.alg, args.signature, args.k, args.metric, ("%.2f" % args.window).replace('.', ''))
		if args.metric == "dot":
			file = name + ".gram"
		else:
			file = name + ".distances"
	if args.alg in ["SEKS"]:
		name = "%s__%s_%d" % (args.dataset, args.alg, args.k)
		file = name + ".gram"
	if args.alg in ["TMAP"]:
		name = "%s__%s%d_%d" % (args.dataset, args.alg, args.S, args.k)
		file = name + ".gram"

	acc, std = train(output_path, args.dataset, file, args.classifier, 100)

	result = {name: [acc, std]}
	save_result(result, output_path, args.dataset, args.classifier)

	# translate = {"tw": "tw", "tgw": "tgw", "tadj": "tadj",
				# "corr": "corr", "SEKS": "SERW", "tgembed": "tgembed"}
	# algorithms = distances + kernels + tgkernels
	# datasets = ["development", "abide"]
	# save_results(output_path, datasets, algorithms, translate)


