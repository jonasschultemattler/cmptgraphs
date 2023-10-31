import sys
import os
import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

from dataloader import load_data


def train_svms(path, datasets, configs):
	accuracies = -1*np.ones((len(configs), len(datasets)))
	variances = -1*np.ones((len(configs), len(datasets)))
	for i, dataset in enumerate(datasets):
		for j, config in enumerate(configs):
			if os.path.exists(path + dataset + "/distances_sample/" + dataset + "_" + config + ".distances"):
				print("training svm for %s %s..." % (dataset, config))

				score, std = train_svm(path + dataset + "/distances_sample/" + dataset + "_" + config + ".distances")
				accuracies[j,i] = score*100
				variances[j,i] = std*100
	return accuracies, variances


def rbf_distance_kernel(dmatrix, gamma=1):
	return np.exp(-gamma*dmatrix**2)


def load_kernel(file, kernel):
	if file.endswith('.gram'):
		K, y = load_data(file)
	elif file.endswith('.distances'):
		m = np.loadtxt(file)
		n = m.shape[0]
		dmatrix, y = m[:,:n], m[:,n]
		if np.isnan(dmatrix).any() or not np.isfinite(dmatrix).all():
			raise ValueError("inf or nan in %s", file)
		if kernel == "rbf":
			dmatrix /= np.max(dmatrix)
			K = rbf_distance_kernel(dmatrix)
		else:
			raise Exception("no such kernel defined")
	return K, y


def load_distances(file):
	if file.endswith('.gram'):
		dmatrix = None # todo
	elif file.endswith('.distances'):
		m = np.loadtxt(file)
		n = m.shape[0]
		dmatrix, y = m[:,:n], m[:,n]
		if np.isnan(dmatrix).any() or not np.isfinite(dmatrix).all():
			raise ValueError("inf or nan in %s", file)
	return dmatrix, y


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


def train_svm(file, kernel="rbf", k=10, l=10):
	D, y = load_distances(file)
	D /= np.max(D)
	# D = norm_matrix(D)
	n = 10

	accuracies, variances = np.zeros(n), np.zeros(n)
	for iteration in range(n):
		print("Iteration %d:" % iteration)
		Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]
		# gammas = [10e-12, 10e-11, 10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10]
		gammas = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10]

		training_accuracies, test_accuracies = np.zeros(k), np.zeros(k)

		outer_cv = KFold(n_splits=k, shuffle=True, random_state=iteration).split(D)
		for i, (train_index, test_index) in enumerate(outer_cv):
			D_train, y_train = D[train_index][:,train_index], y[train_index]
			D_test, y_test = D[test_index][:,train_index], y[test_index]

			inner_cv = KFold(n_splits=l, shuffle=True, random_state=i).split(D_train)
			for train_index_i, test_index_i in inner_cv:
				best_score, best_C, best_gamma = -1, -1, -1
				for c in Cs:
					D_train_i, y_train_i = D_train[train_index_i][:,train_index_i], y_train[train_index_i]
					D_test_i, y_test_i = D_train[test_index_i][:,train_index_i], y_train[test_index_i]

					svc = svm.SVC(kernel='precomputed', C=c)
					for gamma in gammas:
						K_train_i = rbf_distance_kernel(D_train_i, gamma=gamma)
						K_test_i = rbf_distance_kernel(D_test_i, gamma=gamma)
						# K_train_i, K_test_i = norm_matrix(K_train_i), norm_matrix(K_test_i)

						svc.fit(K_train_i, y_train_i)
						score = np.mean(svc.score(K_test_i, y_test_i))
						if score > best_score:
							best_score, best_C, best_gamma = score, c, gamma

			K_train, K_test = rbf_distance_kernel(D_train, gamma=best_gamma), rbf_distance_kernel(D_test, gamma=best_gamma)
			# K_train, K_test = norm_matrix(K_train), norm_matrix(K_test)
			
			svc = svm.SVC(kernel='precomputed', C=best_C)
			svc.fit(K_train, y_train)
			test_accuracies[i] = np.mean(svc.score(K_test, y_test))
			training_accuracies[i] = np.mean(svc.score(K_train, y_train))

			print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s, gamma %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C, best_gamma))

		print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
		print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

		accuracies[iteration] = test_accuracies.mean()
		variances[iteration] = test_accuracies.std()

	print("Overall accuracy: %.1f+-%.1f" % (accuracies.mean()*100, variances.mean()*100))

	return accuracies.mean(), variances.mean()


def get_configs(path, datasets, file_ending):
	configs = set()
	for dataset in datasets:
		n = len(dataset) + 1
		for file_name in os.listdir(path + dataset + "/distances_sample/"):
			if file_name.endswith(file_ending):
				config = file_name[n:-10]
				configs.add(config)
	return sorted(list(configs))


def save_results(path, datasets, configs, accuracies, variances):
	translate = {"optimistic_matching": "$\\sigma_{\\text{opt}}$",
	"sigma*": "$\\sigma^*$", "diagonal_warping": "swp", "optimistic_warping": "owp"}
	with open(path + "accuracies_sample.csv", "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["signature", "init"] + datasets)
		best_scores = np.argmax(accuracies, axis=0)
		for i, config in enumerate(configs):
			signature, init, window = config.split('__')
			row = []
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow([signature[:-10], translate[init[:-5]]] + row)


if __name__ == "__main__":
	path = "../../datasets/"
	file_ending = "window_5.distances"
	datasets = [directory for directory in os.listdir(path) if directory.endswith("ct1") and os.path.exists(path + directory + "/distances_sample")]
	configs = get_configs(path, datasets, file_ending)

	accuracies, variances = train_svms(path, datasets, configs)

	save_results(path, datasets, configs, accuracies, variances)


