import sys
import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold


def load_data(file):
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
	return gram_matrix, labels


def train_svm(path, k=10, l=10, n=10):
	# hs = [0,1,2,3,4,5]
	hs = [0,1,2]
	Cs = [10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3]
	best_h, best_score, best_variance = -1, -1, -1
	
	for h in hs:
		K, y = load_data(path + "_" + str(h) + ".gram")
		K = K[:100,:100]
		y = y[:100]
		print("train svm for %s" % (path + "_" + str(h) + ".gram ..."))

		accuracies, variances = np.zeros(n), np.zeros(n)
		for iteration in range(n):
			print("Iteration %d:" % iteration)

			training_accuracies, test_accuracies = np.zeros(k), np.zeros(k)

			outer_cv = KFold(n_splits=k, shuffle=True, random_state=iteration).split(K)
			for i, (train_index, test_index) in enumerate(outer_cv):
				K_train, y_train = K[train_index][:,train_index], y[train_index]
				K_test, y_test = K[test_index][:,train_index], y[test_index]

				inner_cv = KFold(n_splits=l, shuffle=True, random_state=i).split(K_train)
				for train_index_i, test_index_i in inner_cv:
					best_c_score, best_C = -1, -1
					for c in Cs:
						K_train_i, y_train_i = K_train[train_index_i][:,train_index_i], y_train[train_index_i]
						K_test_i, y_test_i = K_train[test_index_i][:,train_index_i], y_train[test_index_i]

						svc = svm.SVC(kernel='precomputed', C=c)
						svc.fit(K_train_i, y_train_i)
						score = np.mean(svc.score(K_test_i, y_test_i))
						if score > best_c_score:
							best_c_score, best_C = score, c
				
				svc = svm.SVC(kernel='precomputed', C=best_C)
				svc.fit(K_train, y_train)
				test_accuracies[i] = np.mean(svc.score(K_test, y_test))
				training_accuracies[i] = np.mean(svc.score(K_train, y_train))

				# print('    Fold %d: prediction accuracy %.1f, best training accuracy %.1f, C %s' % (i+1, test_accuracies[i]*100, best_score*100, best_C))

			print('    Training accuracy: %.1f+-%.1f%%' % (training_accuracies.mean()*100, training_accuracies.std()*100))
			print('    Prediction accuracy: %.1f+-%.1f%%' % (test_accuracies.mean()*100, test_accuracies.std()*100))

			accuracies[iteration] = test_accuracies.mean()
			variances[iteration] = test_accuracies.std()

		accuracy, variance = accuracies.mean()*100, variances.mean()*100
		print("Overall accuracy: %.1f+-%.1f" % (accuracy, variance))
		if accuracy > best_score:
			best_h, best_score, best_variance = h, accuracy, variance

	return best_score, best_variance


def train_svms(path, datasets, configs):
	accuracies = -1*np.ones((len(configs), len(datasets)))
	variances = -1*np.ones((len(configs), len(datasets)))
	for i, dataset in enumerate(datasets):
		for j, config in enumerate(configs):
			score, std = train_svm(path + dataset + "__" + config)
			accuracies[j,i] = score
			variances[j,i] = std
	return accuracies, variances


def save_results(path, datasets, configs, accuracies, variances):
	with open(path + "accuracies_tgkernel.csv", "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["kernel"] + datasets)
		best_scores = np.argmax(accuracies, axis=0)
		for i, config in enumerate(configs):
			row = [config]
			for j, (acc, var) in enumerate(zip(list(accuracies[i,:]), list(variances[i,:]))):
				if best_scores[j] == i:
					row.append("\\textbf{%.1f}$\\pm$\\tiny %.1f" % (acc, var))
				else:
					row.append("%.1f$\\pm$\\tiny %.1f" % (acc, var))
			writer.writerow(row)



if __name__ == "__main__":
	# datasets = ["infectious_ct1", "highschool_ct1", "tumblr_ct1", "dblp_ct1", "facebook_ct1"]
	# datasets = ["infectious_ct2", "highschool_ct2", "tumblr_ct2", "dblp_ct2", "facebook_ct2"]
	datasets = ["development"]
	configs = ["SEKS", "SEWL"]
	path = "tgkernel/release/"

	accuracies, variances = train_svms(path, datasets, configs)

	save_results("output/brains/development/", datasets, configs, accuracies, variances)

