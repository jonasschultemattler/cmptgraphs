import os

import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import pairwise, accuracy_score


def load_txt(file, number):
	with open(file) as fp:
		lines = fp.readlines()
		n = len(lines)
		gram_matrix = np.zeros((n, n))
		labels = np.zeros(n)
		for i, line in enumerate(lines):
			line_split = line.split(':')
			labels[i] = int(line_split[0].split(' ')[0])
			for j, column in enumerate(line_split[2:]):
				gram_matrix[i][j] = float(column.split(' ')[0])
	return gram_matrix[:number,:number], labels[:number]


def load(file, number):
	m = np.loadtxt(file)
	n = m.shape[0]
	matrix, y = m[:,:n], m[:,n]
	if np.isnan(matrix).any() or not np.isfinite(matrix).all():
		raise ValueError("inf or nan in %s", file)
	return matrix[:number,:number], y[:number]


def kernel_distance(kernel):
	n = kernel.shape[0]
	D = np.empty((n, n))
	for i in range(n):
		for j in range(i,n):
			D[i,j] = D[j,i] = kernel[i,i] - 2*kernel[i,j] + kernel[j,j]
	return np.sqrt(D)


def linear_distance_kernel(dmatrix):
	n = dmatrix.shape[0]
	i_origin = 0
	K = np.empty((n, n))
	for i in range(n):
		for j in range(i,n):
			K[i,j] = K[j,i] = -0.5*(dmatrix[i,j]**2 - dmatrix[i,i_origin]**2 - dmatrix[j,i_origin]**2)
	# K = -0.5*(dmatrix**2 - (dmatrix[i_origin,:])**2 - (dmatrix[:,i_origin])**2)
	return K


def rbf_distance_kernel(dmatrix, gamma=1):
	return np.exp(gamma*dmatrix**2)



def train(output_path, dataset, name, classifier, number):
	print("training " + name + "...")
	if name.endswith(".gram"):
		if dataset in name:
			K, y = load_txt(os.path.join(output_path, dataset, name), number)
		else:
			K, y = load(os.path.join(output_path, dataset, name), number)
		D = kernel_distance(K)
	else:
		D, y = load(os.path.join(output_path, dataset, name), number)
		K = linear_distance_kernel(D)

	iterations, folds = 5, 5
	if classifier == "SVM_linear":
		acc, std = train_svm(K, y, iterations=iterations, folds=folds)
	elif classifier == "SVM_rbf":
		acc, std = train_svm_rbf(D, y, iterations=iterations, folds=folds)
	elif classifier == "kNN":
		acc, std = train_kNN(D, y, iterations=iterations, folds=folds)

	return acc, std


def train_svm(K, y, iterations=10, folds=10):
	Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]

	accuracies, variances = np.zeros(iterations), np.zeros(iterations)
	for iteration in range(iterations):
		print("Iteration %d:" % iteration)

		training_accuracies, test_accuracies = np.zeros(folds), np.zeros(folds)

		cv = KFold(n_splits=folds, shuffle=True, random_state=iteration).split(K)
		best_c_score, best_C = -1, -1

		for i, (train_index, test_index) in enumerate(cv):
			K_train, y_train = K[train_index][:,train_index], y[train_index]
			K_test, y_test = K[test_index][:,train_index], y[test_index]

			for C in Cs:
				svc = svm.SVC(kernel='precomputed', C=C)
				svc.fit(K_train, y_train)
				score = np.mean(svc.score(K_test, y_test))
				if score > best_c_score:
					best_c_score, best_C = score, C
			
			svc = svm.SVC(kernel='precomputed', C=best_C)
			svc.fit(K_train, y_train)
			training_accuracies[i] = np.mean(svc.score(K_train, y_train))
			test_accuracies[i] = np.mean(svc.score(K_test, y_test))

			# print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C))

		print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
		print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

		accuracies[iteration] = test_accuracies.mean()
		variances[iteration] = test_accuracies.std()

	accuracy, variance = accuracies.mean()*100, variances.mean()*100

	print("Overall accuracy: %.1f+-%.1f%%" % (accuracy, variance))

	return accuracy, variance


def train_svm_rbf(D, y, iterations=10, folds=10):
	accuracies, variances = np.zeros(iterations), np.zeros(iterations)
	for iteration in range(iterations):
		print("Iteration %d:" % iteration)
		Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]
		gammas = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10]

		training_accuracies, test_accuracies = np.zeros(folds), np.zeros(folds)

		cv = KFold(n_splits=folds, shuffle=True, random_state=iteration).split(D)
		for i, (train_index, test_index) in enumerate(cv):
			D_train, y_train = D[train_index][:,train_index], y[train_index]
			D_test, y_test = D[test_index][:,train_index], y[test_index]
			
			best_score, best_C, best_gamma = -1, -1, -1
			for c in Cs:
				svc = svm.SVC(kernel='precomputed', C=c)
				for gamma in gammas:
					K_train = rbf_distance_kernel(D_train, gamma=gamma)
					K_test = rbf_distance_kernel(D_test, gamma=gamma)
					svc.fit(K_train, y_train)
					score = np.mean(svc.score(K_test, y_test))
					if score > best_score:
						best_score, best_C, best_gamma = score, c, gamma

			K_train, K_test = rbf_distance_kernel(D_train, gamma=best_gamma), rbf_distance_kernel(D_test, gamma=best_gamma)
			svc = svm.SVC(kernel='precomputed', C=best_C)
			svc.fit(K_train, y_train)
			test_accuracies[i] = np.mean(svc.score(K_test, y_test))
			training_accuracies[i] = np.mean(svc.score(K_train, y_train))

			# print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s, gamma %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C, best_gamma))

		print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
		print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

		accuracies[iteration] = test_accuracies.mean()
		variances[iteration] = test_accuracies.std()

	accuracy, variance = accuracies.mean()*100, variances.mean()*100
	print("Overall accuracy: %.1f+-%.1f%%" % (accuracy, variance))

	return accuracy, variance


def train_kNN(D, y, iterations=10, folds=10):
	outer_cv = KFold(n_splits=iterations, shuffle=True)
	inner_cv = KFold(n_splits=folds, shuffle=True)

	ks = list(range(1,10))
	hyper_params = dict(n_neighbors=ks)

	knn = KNeighborsClassifier(metric='precomputed')

	grid = GridSearchCV(estimator=knn, param_grid=hyper_params, cv=inner_cv, refit=True)
	grid.fit(D, y)

	accuracies = cross_val_score(grid, D, y, cv=outer_cv)

	acc, std = accuracies.mean()*100, accuracies.std()*100
	print('Accuracy: %.1f+-%.1f%%  hyper_params: %s' % (acc, std, grid.best_params_))

	return acc, std


