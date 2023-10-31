import numpy as np
import os



class TemporalGraph():
	def __init__(self, tadj):
		self.tadj = tadj

	def save(self, path):
		np.save(path, self.tadj)



class LabeledTemporalGraph(TemporalGraph):
	def __init__(self, tadj, tlabels):
		super().__init__(tadj)
		self.tlabels = tlabels

	# def save(self, path):
	# 	print(self.tadj.shape)
	# 	print(self.tlabels.shape)
	# 	print(np.array([self.tadj, self.tlabels]).shape)
	# 	np.save(path, np.array([self.tadj, self.tlabels]))


class DisseminationDataloader():
	def __init__(self, name, dir_path):
		self.name = name
		self.dir_path = dir_path
		self.path = os.path.join(dir_path, self.name)


	def tedges2tadj(tedges, T, n):
		tadj = np.zeros((T,n,n), dtype=bool)
		for t, v, w in tedges:
			tadj[t,v,w] = tadj[t,w,v] = True
		return tadj

	def strip_node_labels(line, lifetime):
		node_labels = np.zeros(lifetime)
		line_split = line.split(',')
		for j in range(0, len(line_split), 2):
			t, label = int(line_split[j]), int(line_split[j+1])
			node_labels[t-1:] = label
		return node_labels


	def loadtxt(self, number):
		all_edges = []
		with open(os.path.join(self.path, self.name + "_A.txt")) as graphs_edges:
			for line in graphs_edges:
				v, w = line.split(',')
				v, w = int(v), int(w)
				all_edges.append((v, w))

		all_tedges = []
		graphs_temporal_edges = open(os.path.join(self.path, self.name + "_edge_attributes.txt"))
		for i, line in enumerate(graphs_temporal_edges):
			t = int(line)
			v, w = all_edges[i]
			all_tedges.append((t,v,w))

		graphs, vertex_graph = set(), dict()
		graphs_vertices = open(os.path.join(self.path, self.name + "_graph_indicator.txt"))
		for i, line in enumerate(graphs_vertices):
			graphs.add(int(line)-1)
			vertex_graph.update({i+1: int(line)-1})

		graph_vertices = {graph: set() for graph in graphs}
		graph_lifetimes = {graph: 0 for graph in graphs}
		for t, v, w in all_tedges:
			g1, g2 = vertex_graph[v], vertex_graph[w]
			assert(g1 == g2)
			graph_vertices[g1].add(v)
			graph_vertices[g1].add(w)
			graph_lifetimes[g1] = max(t+1, graph_lifetimes[g1])

		graph_vertices_index = {}
		for graph, vertices in graph_vertices.items():
			vertices_index = {v: i for i, v in enumerate(vertices)}
			graph_vertices_index.update({graph: vertices_index})

		graph_tedges = {graph: [] for graph in graphs}
		for t, v, w in all_tedges:
			g = vertex_graph[v]
			graph_tedges[g].append((t-1,graph_vertices_index[g][v],graph_vertices_index[g][w]))

		tadjs = {}
		for graph, tedges in graph_tedges.items():
			T, n = graph_lifetimes[graph], len(graph_vertices[graph])
			tadjs.update({graph: tedges2tadj(tedges, T, n)})

		node_labels = {g: np.zeros((graph_lifetimes[g], len(graph_vertices[g])), dtype=bool) for g in graphs}
		temporal_node_labels = open(os.path.join(self.path, self.name + "_node_labels.txt"))
		for v, line in enumerate(temporal_node_labels):
			g = vertex_graph[v+1]
			i = graph_vertices_index[g][v+1]
			node_labels[g][:,i] = strip_node_labels(line, graph_lifetimes[g])

		graph_labels = {graph: False for graph in graphs}
		graph_label_file = open(os.path.join(self.path, self.name + "_graph_labels.txt"))
		for g, line in enumerate(graph_label_file):
			graph_labels[g] = bool(int(line))

		tgraphs, labels = [], []
		for g in graphs:
			tgraphs.append(LabeledTemporalGraph(tadjs[g], node_labels[g]))
			labels.append(graph_labels[g])
		
		return tgraphs[:number], labels[:number]


	def load(self):
		files = [f for f in os.listdir(self.path) if f.startswith(self.name + "_t") and f.endswith(".npy")]
		files = sorted(files, key=lambda f: int(f.split('t')[-1][:-4]))
		tgraphs = [np.load(os.path.join(self.path, f)) for f in files]
		labels = np.load(os.path.join(self.path, "labels.npy"))
		return tgraphs, labels


	def save(self, tgraphs, labels):
		for i, g in enumerate(tgraphs):
			g.save(os.path.join(self.path, "%s_t%d.npy" % (self.name, i)))
		np.save(os.path.join(self.path, "labels.npy"), np.array(labels))


	def savetxt(self, tgraphs, labels):
		edges_file = open(os.path.join(self.path, self.name + "_A2.txt"), 'w')
		tedges_file = open(os.path.join(self.path, self.name + "_edge_attributes2.txt"), 'w')
		c = 0
		for tgraph in tgraphs:
			tadj = tgraph.tadj
			T, n, _ = tadj.shape
			for i in range(n):
				for t in range(T):
					for j in range(n):
						if tadj[t][i][j]:
							edges_file.write("%d, %d\n" % (c+i+1, c+j+1))
							tedges_file.write("%d\n" % (t+1))
			c += n
		graph_file = open(os.path.join(self.path, self.name + "_graph_indicator2.txt"), 'w')
		for g, tgraph in enumerate(tgraphs):
			for i in range(tgraph.tadj.shape[1]):
				graph_file.write("%d\n" % (g+1))
		labels_file = open(os.path.join(self.path, self.name + "_graph_labels2.txt"), 'w')
		for label in labels:
			labels_file.write("%d\n" % label)
		node_labels_file = open(os.path.join(self.path, self.name + "_node_labels2.txt"), 'w')
		for tgraph in tgraphs:
			for v in range(tgraph.tlabels.shape[1]):
				label = tgraph.tlabels[0,v]
				line = "0, %d" % label
				for t in range(1, tgraph.tlabels.shape[0]):
					if tgraph.tlabels[t,v] != label:
						label = tgraph.tlabels[t,v]
						line += ", %d, %d" % (t+1, tgraph.tlabels[t,v])
				node_labels_file.write(line + "\n")




def load_static_graphs(path, name):
	all_edges = []
	with open(os.path.join(path, name + "_A.txt")) as graphs_edges:
		for line in graphs_edges:
			v, w = line.split(',')
			v, w = int(v), int(w)
			all_edges.append((v, w))

	graphs, vertex_graph = set(), dict()
	graphs_vertices = open(os.path.join(path, name + "_graph_indicator.txt"))
	for i, line in enumerate(graphs_vertices):
		graphs.add(int(line)-1)
		vertex_graph.update({i+1: int(line)-1})

	graph_vertices = {graph: set() for graph in graphs}
	for v, w in all_edges:
		g1, g2 = vertex_graph[v], vertex_graph[w]
		assert(g1 == g2)
		graph_vertices[g1].add(v)
		graph_vertices[g1].add(w)

	graph_vertices_index = {}
	for graph, vertices in graph_vertices.items():
		vertices_index = {v: i for i, v in enumerate(vertices)}
		graph_vertices_index.update({graph: vertices_index})

	graph_edges = {graph: [] for graph in graphs}
	for v, w in all_edges:
		g = vertex_graph[v]
		graph_edges[g].append((t-1,graph_vertices_index[g][v],graph_vertices_index[g][w]))

	return graph_edges




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


