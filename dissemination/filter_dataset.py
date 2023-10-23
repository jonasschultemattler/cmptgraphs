import os
import numpy as np
from dataloader import DisseminationDataloader



def filter_dataset(tgraphs, labels):
	res_tgraphs, res_labels = [], []
	for tgraph, label in zip(tgraphs, labels):
		vertices = tgraph.tadj.shape[1]
		infections = np.sum(tgraph.tlabels[-1])
		# print(infections/vertices)
		if infections >= vertices/2:
			res_tgraphs.append(tgraph)
			res_labels.append(label)
	return res_tgraphs, res_labels



if __name__ == "__main__":
	datasets = ["mit", "infectious", "tumblr", "highschool", "dblp", "facebook"]
	# datasets = ["dblp"]
	dataset_path = os.path.join("..", "datasets", "dissemination")
	n = 1000

	for task in ["_ct1", "_ct2"]:
		print(task)
		for dataset in datasets:
			print(dataset)
			dataloader = DisseminationDataloader(dataset + task, dataset_path)
			tgraphs, labels = dataloader.loadtxt(n)
			tgraphs2, labels2 = filter_dataset(tgraphs, labels)
			dataloader.savetxt(tgraphs2[:100], labels2[:100])
			# print(len(tgraphs))
		print()
		print()
	