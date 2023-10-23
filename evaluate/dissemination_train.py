import sys
import os
import csv
import json
import argparse
import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import pairwise, accuracy_score

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
	parser.add_argument("--alg", choices=("dtgw", "SEKS", "SEWL", "LGKS", "LGWL", "tkg10", "tkg11", "tkgw"), default="dtgw", help="algorithm to compute distances")
	# parser.add_argument("--signature", choices=("degree", "degree2", "degree3", "degree4", "neighbors", "neighbors2", "neighbors3"), default="neighbors", help="vertex signature")
	parser.add_argument("--signature", choices=("subtrees", "walks", "neighbors", "subtrees2"), default="subtrees", help="vertex signature")
	parser.add_argument("--init",
		choices = ("diagonal_warping", "optimistic_warping", "sigma*", "optimistic_matching"),
		default = "diagonal_warping", help = "Initialization to  use for the heuristic")
	parser.add_argument("--metric", choices=("l1", "l2", "dot"), default="l1", help="metric norm")
	parser.add_argument("--window", type=float, default=0.2, help="relative window size for time warping")
	parser.add_argument("--k", type=int, default=2, help="k depth subtree; random walk steps")
	parser.add_argument("--dcost", type=int, default=0, help="pay 0, f(v) or f(v)/|n-m| for not matched vertices")
	# parser.add_argument("--pay", type=bool, default=False, help="pay for not matched vertices")
	parser.add_argument("--classifier", choices=("SVM_linear", "SVM_rbf", "kNN"), default="SVM_linear", help="classifier")
	parser.add_argument("--number", type=int, default=100, help="number of graphs to train")
	parser.add_argument("--iterations", type=int, default=5, help="max iterations per heuristic computation")
	args = parser.parse_args()

	output_path = "../output/dissemination"

	number = args.number
	if args.alg == "dtgw":
		print(args.dataset)
		if "mit" in args.dataset:
			number = min(number, 89)
		name = "dtgw_%s%d_c%d_%s_%s_w%s_i%d_n%d" % (args.signature, args.k, args.dcost, args.init, args.metric, ("%.2f" % args.window).replace('.', ''), args.iterations, number)
		if args.metric == "dot":
			file = name + ".gram"
		else:
			file = name + ".distances"
	elif args.alg in ["SEKS", "SEWL", "LGKS", "LGWL", "tkg10", "tkg11", "tkgw"]:
		name = "%s__%s_%d" % (args.dataset, args.alg, args.k)
		file = name + ".gram"
	else:
		pass

	acc, std = train(output_path, args.dataset, file, args.classifier, number)

	result = {name: [acc, std]}
	save_result(result, output_path, args.dataset, args.classifier)



