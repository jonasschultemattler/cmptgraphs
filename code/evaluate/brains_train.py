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



