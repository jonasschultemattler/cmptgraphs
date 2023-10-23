import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from dissemination.dataloader import DisseminationDataloader


def load_tgkernel(file):
	with open(file) as fp:
		lines = fp.readlines()
		n = len(lines)
		gram_matrix = np.zeros((n, n))
		for i, line in enumerate(lines):
			for j, column in enumerate(line.split(':')[2:]):
				gram_matrix[i][j] = float(column.split(' ')[0])
	return gram_matrix

def kernel_to_distance(gram_matrix):
	distance_matrix = np.zeros(gram_matrix.shape)
	for i in range(gram_matrix.shape[0]):
		for j in range(gram_matrix.shape[1]):
			distance_matrix[i][j] = np.sqrt(gram_matrix[i][i] - 2*gram_matrix[i][j] + gram_matrix[j][j])
	return distance_matrix


def index_order_infections(labels, tgraphs):
	labels = labels.astype(bool)
	class1_indices = np.where(labels)[0]
	class2_indices = np.where(~labels)[0]
	infections = np.array([np.sum(tgraph.tlabels[-1]) for tgraph in tgraphs])
	sorted_class1_indices = class1_indices[np.argsort(infections[class1_indices])]
	sorted_class2_indices = class2_indices[np.argsort(infections[class2_indices])]
	order = np.hstack([sorted_class1_indices, sorted_class2_indices])
	return order


def index_order_vertex(labels, tgraphs):
	labels = labels.astype(bool)
	class1_indices = np.where(labels)[0]
	class2_indices = np.where(~labels)[0]
	vertices = np.array([tgraph.tadj.shape[1] for tgraph in tgraphs])
	sorted_class1_indices = class1_indices[np.argsort(vertices[class1_indices])]
	sorted_class2_indices = class2_indices[np.argsort(vertices[class2_indices])]
	# order = np.hstack([class1_indices, class2_indices])
	order = np.hstack([sorted_class1_indices, sorted_class2_indices])
	return order


def sort_matrix(matrix, labels, tgraphs):
	n = matrix.shape[0]
	sorted_matrix = np.empty((n,n))
	# order = index_order_vertex(labels, tgraphs)
	order = index_order_infections(labels, tgraphs)
	a,b = np.triu_indices(n)
	sorted_matrix[a,b] = matrix[ [order[i] for i in a], [order[j] for j in b] ]
	sorted_matrix[b,a] = sorted_matrix[a,b]
	return sorted_matrix


if __name__ == "__main__":
	cm = 1/2.54
	W = 7.84
	plt.rcParams.update({
    	'figure.figsize': (W*cm, W*cm),
    	'font.size': 11,
    	'axes.labelsize': 11,
    	'xtick.labelsize': 10,
    	# 'legend.fontsize': 12,
    	'axes.titlesize': 11,
    	'font.family': 'lmodern',
    	'text.usetex': True,
    	'text.latex.preamble': r'\usepackage{lmodern}',
    	'figure.autolayout': True
	})

	dataset = "facebook"
	task = "_ct1"
	file = "dtgw_subtrees2_c2_diagonal_warping_l1_w020_i5_n100.distances"

	output_path = os.path.join("..", "output", "dissemination", dataset + task)
	dataset_path = os.path.join("..", "datasets", "dissemination")

	if file.endswith(".distances"):
		matrix = np.loadtxt(os.path.join(output_path, file))
	if file.endswith(".gram"):
		matrix = kernel_to_distance(load_tgkernel(os.path.join(output_path, file)))

	n = 100

	gram_matrix, labels = matrix[:,:-1][:n,:n], matrix[:,-1][:n]
	
	dataloader = DisseminationDataloader(dataset + task, dataset_path)
	tgraphs, _ = dataloader.loadtxt(n)

	sorted_matrix = sort_matrix(gram_matrix, labels, tgraphs)
	
	fig, axes = plt.subplots(1)
	pos = axes.imshow(sorted_matrix)
	# fig.colorbar(pos, ax=axes)

	infections = np.array([np.sum(tgraph.tlabels[-1]) for tgraph in tgraphs])
	class1_indices = np.where(labels.astype(bool))[0]
	class2_indices = np.where(~labels.astype(bool))[0]
	sinfections1 = np.array(sorted(infections[class1_indices]))
	sinfections2 = np.array(sorted(infections[class2_indices]))
	sorted_infections = np.hstack([sinfections1, sinfections2])
	print(sorted_infections)

	limits = [3, 15, 45, 50]
	xticks1 = np.array([np.where(sinfections1 <= limits[i])[0][-1] for i in range(len(limits))])
	xticks2 = np.array([np.where(sinfections2 <= limits[i])[0][-1] for i in range(len(limits))])
	axes.set_xticks(np.hstack([xticks1, n//2 + xticks2]))
	axes.set_xticklabels(np.hstack([sinfections1[xticks1], sinfections2[xticks2]]))
	axes.set_xlabel("Total number of infections")
	axes.set_yticks([25, 75])
	axes.set_yticklabels(["Class 1", "Class 2"], rotation=90, va="center")
	axes.set_title("Dtgw-distances for \\textsc{Facebook} task 1")
	
	axes.plot([49.5,49.5], [-0.5,99.5], color='white', linewidth=1)
	axes.plot([-0.5,99.5], [49.5,49.5], color='white', linewidth=1)
	fig.savefig(os.path.join(output_path, file.split('.')[0] + "_heatmap.pdf"), dpi=1000)




	dataset = "tumblr"
	task = "_ct1"
	file = "dtgw_subtrees0_c2_diagonal_warping_l2_w020_i5_n100.distances"

	output_path = os.path.join("..", "output", "dissemination", dataset + task)
	dataset_path = os.path.join("..", "datasets", "dissemination")

	if file.endswith(".distances"):
		matrix = np.loadtxt(os.path.join(output_path, file))
	if file.endswith(".gram"):
		matrix = kernel_to_distance(load_tgkernel(os.path.join(output_path, file)))

	n = 100

	gram_matrix, labels = matrix[:,:-1][:n,:n], matrix[:,-1][:n]
	
	dataloader = DisseminationDataloader(dataset + task, dataset_path)
	tgraphs, _ = dataloader.loadtxt(n)

	sorted_matrix = sort_matrix(gram_matrix, labels, tgraphs)
	
	fig, axes = plt.subplots(1)
	pos = axes.imshow(sorted_matrix)
	# fig.colorbar(pos, ax=axes)

	infections = np.array([np.sum(tgraph.tlabels[-1]) for tgraph in tgraphs])
	class1_indices = np.where(labels.astype(bool))[0]
	class2_indices = np.where(~labels.astype(bool))[0]
	sinfections1 = np.array(sorted(infections[class1_indices]))
	sinfections2 = np.array(sorted(infections[class2_indices]))
	sorted_infections = np.hstack([sinfections1, sinfections2])
	print(sorted_infections)

	limits = [3, 15, 25, 40]
	xticks1 = np.array([np.where(sinfections1 <= limits[i])[0][-1] for i in range(len(limits))])
	xticks2 = np.array([np.where(sinfections2 <= limits[i])[0][-1] for i in range(len(limits))])
	axes.set_xticks(np.hstack([xticks1, n//2 + xticks2]))
	axes.set_xticklabels(np.hstack([sinfections1[xticks1], sinfections2[xticks2]]))
	axes.set_xlabel("Total number of infections")
	axes.set_yticks([25, 75])
	axes.set_yticklabels(["Class 1", "Class 2"], rotation=90, va="center")
	axes.set_title("Dtgw-distances for \\textsc{Tumblr} task 1")
	
	axes.plot([49.5,49.5], [-0.5,99.5], color='white', linewidth=1)
	axes.plot([-0.5,99.5], [49.5,49.5], color='white', linewidth=1)
	fig.savefig(os.path.join(output_path, file.split('.')[0] + "_heatmap.pdf"), dpi=1000)



