import sys
import os
import csv
import numpy as np
from sklearn import svm
from dataloader import load_data


# def linear_distance_kernel(dmatrix):
	# return -(dmatrix**2 - 2*np.abs(dmatrix)**2)/2

def rbf_distance_kernel(dmatrix, gamma=1):
	return np.exp(-gamma*dmatrix**2)


def train_svm(file, kernel="rbf"):
	if file.endswith('.gram'):
		Ktrain, ytrain = load_data(file)
	elif file.endswith('.distances'):
		m = np.loadtxt(file)
		n = m.shape[0]
		dmatrix, ytrain = m[:,:n], m[:,n]
		# ytrain[ytrain==0] = -1
		if np.isnan(dmatrix).any() or not np.isfinite(dmatrix).all():
			return -1/100
		if kernel == "rbf":
			dmatrix /= np.max(dmatrix)
			Ktrain = rbf_distance_kernel(dmatrix)
			# Ktrain -= Ktrain.mean()
			# Ktrain /= Ktrain.std()

	mysvm = svm.SVC(kernel='precomputed').fit(Ktrain, ytrain)
	score = mysvm.score(Ktrain, ytrain)
	print("done.\n score: %.1f%%" % (score*100))
	return score



if __name__ == "__main__":
	path = "../../datasets/"
	datasets = [directory for directory in os.listdir(path) if os.path.exists(path + directory + "/distances")]
	configs = set()
	for dataset in datasets:
		n = len(dataset) + 1
		for file_name in os.listdir(path + dataset + "/distances/"):
			if file_name.endswith("window_5.distances"):
				config = file_name[n:-10]
				configs.add(config)
	configs = sorted(list(configs))
	accuracies = -1*np.ones((len(configs), len(datasets)))
	for i, dataset in enumerate(datasets):
		for j, config in enumerate(configs):
			if os.path.exists(path + dataset + "/distances/" + dataset + "_" + config + ".distances"):

				print("training svm for %s %s..." % (dataset, config))
				score = train_svm(path + dataset + "/distances/" + dataset + "_" + config + ".distances")
				accuracies[j,i] = score*100
	translate = {"optimistic_matching": "$\\sigma_{\\text{opt}}$",
	"sigma*": "$\\sigma^*$", "diagonal_warping": "swp", "optimistic_warping": "owp"}
	with open(path + "accuracies_window.csv", "w") as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(["signature", "init"] + datasets)
		for j, config in enumerate(configs):
			signature, init, window = config.split('__')
			writer.writerow([signature[:-10], translate[init[:-5]]] + list(accuracies[j,:]))


