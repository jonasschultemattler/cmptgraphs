"""Module for common functionality when dealing with temporal graphs.

Within this module, a "time-edge"
refers to a 3-tuple of the form
"(layer, vertex1, vertex2)",
which either
* defines a layer, in which case »vertex1« and »vertex2« are None, or
* defines a layer and a vertex, in which case »vertex2« is None, or
* defines a layer, two vertices and an edge between them, in which case no entry is None.

A "time-edge file" is a text file containing in each line
a time-edge as space-separated list of values, where None-entries are omitted.
Any empty lines or lines starting with "#" are ignored.
"""


import collections
import collections.abc

import numpy as np
import graph_tool as gt


class ParsingError(Exception):
	pass


class ConfigurationError(Exception):
	pass


class TemporalGraph:
	"""A temporal graph.
	The layers and vertices are always numbered 0, 1, ...
	"""
	def __init__(self, lifetime, vertices=0):
		self.lifetime = lifetime
		self._graph = gt.Graph(directed=False)
		if vertices:
			self._graph.add_vertex(vertices)
		self._layerprop = self._graph.edge_properties["layers"] = self._graph.new_edge_property("vector<int>")
		self._layercache = {}

	@classmethod
	def from_timeedges(cls, timeedges):
		"""Constructs a temporal graph from a time-edge iterable.
		All layers and vertices are required to be non-negative integers
		and it is assumed they are numbered consecutively starting from 0
		(gaps will automatically be filled).
		"""
		vertices = 0
		lifetime = 0
		edgetimes = collections.defaultdict(list)
		for layer,v,w in timeedges:
			lifetime = max(lifetime, layer+1)
			if v is not None:
				if w is not None and w > v:
					v, w = w, v
				vertices = max(vertices, v+1)
				if w is not None:
					edgetimes[v,w].append(layer)
		tg = cls(lifetime=lifetime+1, vertices=vertices)
		tg._graph.add_edge_list(
			((v, w, times) for ((v, w), times) in edgetimes.items()),
			eprops = [tg._layerprop]
		)
		return tg

	def _invalidate_layer_cache(self, layer):
		self._layercache.pop(layer, None)

	def has_timeedge(self, time, v, w):
		edge = self._graph.edge(v, w)
		return edge and time in self._layerprop[edge]

	def add_timeedge(self, time, v, w):
		edge = self._graph.edge(v, w)
		if edge is None:
			edge = self._graph.add_edge(v, w)
			self._layerprop[edge] = [time]
		elif time not in self._layerprop[edge]:
			self._layerprop[edge].append(time)
		self._invalidate_layer_cache(time)

	def add_edge(self, v, w, times):
		edge = self._graph.edge(v, w)
		if edge is None:
			edge = self._graph.add_edge(v, w)
			self._layerprop[edge] = times
			for time in times:
				self._invalidate_layer_cache(time)
		else:
			raise ValueError("Edge already exists")

	def delete_timeedge(self, time, v, w, raise_on_missing=True):
		edge = self._graph.edge(v, w)
		for i, t in enumerate(self._layerprop[edge]):
			if t == time:
				del self._layerprop[edge][i]
				self._invalidate_layer_cache(time)
				if not self._layerprop[edge]:
					self._graph.remove_edge(edge)
				return
		if raise_on_missing:
			raise ValueError("Edge not found")

	def delete_edge(self, v, w, raise_on_missing=True):
		"""Deletes the edge between v and w from all layers."""
		edge = self._graph.edge(v, w)
		if edge:
			self._graph.remove_edge(edge)
		elif raise_on_missing:
			raise ValueError("Edge not found")

	def add_vertex(self, num=1):
		return self._graph.add_vertex(num)

	def layer(self, time):
		if time not in self._layercache:
			# caching is important as it prevents the result from being garbage collected
			# immediately if the caller does not store it
			filter_prop = self._graph.new_edge_property("bool")
			filter_fn = lambda times: time in times
			gt.map_property_values(self._layerprop, filter_prop, filter_fn)
			self._layercache[time] = gt.GraphView(self._graph, efilt=filter_prop)
		return self._layercache[time]

	def num_vertices(self):
		return self._graph.num_vertices()

	def get_vertices(self):
		return self._graph.get_vertices()

	def vertices(self):
		return self._graph.vertices()

	def vertex(self, *args, **kwargs):
		return self._graph.vertex(*args, **kwargs)

	def union_graph(self):
		return self._graph

	def timeedges(self, sorted=False):
		"""Returns a generator producing all time-edges.
		If »sorted« is true, the result is sorted by time (ascending).
		"""
		if sorted:
			for time in range(self.lifetime):
				layer = self.layer(time)
				yield from ((time, int(e.source()), int(e.target())) for e in layer.edges())
		else:
			for edge in self._graph.edges():
				v, w = int(edge.source()), int(edge.target())
				yield from ((time, v, w) for time in self._layerprop[edge])

	def edge_times(self, v, w):
		"""Returns the list of times at which the edge between v and w exists.
		Returns an empty list if there is no edge between v and w.
		"""
		edge = self._graph.edge(v, w)
		if edge is None:
			return []
		return list(self._layerprop[edge])


def iter_timeedge_file(infile, raise_on_trailing_data=False):
	"""Given a time-edge file,
	this method returns a time-edge iterator
	over the file’s contents.

	If »raise_on_trailing_data« is False,
	lines containing more than three values do not
	cause a ValueError but the trailing items are ignored.
	"""
	for line in infile:
		line = line.strip()
		if not line or line[0] == "#":
			continue
		items = line.split()
		layer = items[0]
		v, w = None, None
		if len(items) >= 2:
			v = items[1]
			if len(items) >= 3:
				w = items[2]
				if len(items) > 3 and raise_on_trailing_data:
					raise ValueError("Can not parse line (too many values): »{}«".format(line))
		yield layer, v, w


def map_timeedges(timeedges, vertex_map=int, layer_map=int):
	"""Iterates through a time-edge iterable,
	mapping each vertex through »vertex_map«
	and each layer through »layer_map«.
	"""
	for layer, v, w in timeedges:
		layer = layer_map(layer)
		if v is not None:
			v = vertex_map(v)
			if w is not None:
				w = vertex_map(w)
		yield layer, v, w


def read_timeedges(infile):
	"""Reads time-edges from a file
	and indexes layers and vertices with integers

	Returns:
	--------
	timeedges : iterable of time-edges
		Time-edges where all layers and vertices are
		denominated with 0, 1, ...
	layer_indexing : Indexing
		Indexing of the layer names
	vertex_indexing : Indexing
		Indexing of the vertex names
	"""
	vertex_indexing = Indexing()
	layer_indexing = Indexing()
	timeedges = map_timeedges(
		iter_timeedge_file(infile),
		vertex_map = vertex_indexing.index,
		layer_map = layer_indexing.index
	)
	return timeedges, layer_indexing, vertex_indexing


def write_timeedges(outfile, timeedges, layer_map=str, vertex_map=str):
	"""Writes time-edges to a file.

	Arguments:
	----------
	outfile : file-like object
		File for writing
	timeedges : iterable of time-edges
		Time-edges to write
	layer_map : function mapping layer to string
		Function to convert layers to strings upon writing
	vertex_map : function mapping vertex to string
		Function to convert vertices to strings upon writing
	"""
	for layer, v, w in timeedges:
		outfile.write(layer_map(layer))
		if v is not None:
			outfile.write(" " + vertex_map(v))
			if w is not None:
				outfile.write(" " + vertex_map(w))
		outfile.write("\n")


def read_temporal_graph(infile):
	"""Reads a temporal graph form a file.

	Returns:
	--------
	ltg : LabeledTGraph
		Resulting temporal graph with layer and vertex labels
	"""
	timeedges, layer_indexing, vertex_indexing = read_timeedges(infile)
	tg = TemporalGraph.from_timeedges(timeedges)
	return LabeledTGraph(tg, layer_indexing, vertex_indexing)


class Indexing(collections.abc.Mapping):
	"""An indexing of a collection of hashable objects
	with the non-negative integers.
	"""

	def __init__(self):
		self._indices = {}
		self._objects = []


	@property
	def inverse_map(self):
		return self._indices
	

	def index(self, obj):
		"""If »obj« already has an assigned index,
		return that index.
		Otherwise assign the lowest unused index to »obj«
		and return it.
		""" 
		try:
			return self._indices[obj]
		except KeyError:
			n = len(self._objects)
			self._indices[obj] = n
			self._objects.append(obj)
			return n

	def find(self, obj):
		"""Returns the index assigned to »obj«.
		Raises ValueError if »obj« has no assigned index.
		"""
		try:
			return self._indices[obj]
		except KeyError:
			raise ValueError("No index exists for »{}«".format(obj))

	def lookup(self, index, default=None):
		"""Return the object that was assigned
		the given index »index« or »default« if 
		no such object exists.
		"""
		if index < 0:
			return default
		try:
			return self._objects[index]
		except (IndexError, TypeError):
			return default


	def __getitem__(self, index):
		"""Return the object that was assigned
		the given index »index«.
		Raises IndexError if no such object exists.
		"""
		if index < 0:
			raise IndexError()
		try:
			return self._objects[index]
		except (IndexError, TypeError):
			raise IndexError()


	def __iter__(self):
		"""Iterator over Indices."""
		return iter(range(len(self._objects)))

	def __len__(self):
		return len(self._objects)


class IndexedTGraph:
	def __init__(self, tgraph, layer_index=None, vertex_index=None):
		self.tgraph = tgraph
		self.layer_index = layer_index
		self.vertex_index = vertex_index


class LabeledTGraph(TemporalGraph):
	def __init__(self, lifetime, vertices=0, label=None):
		super().__init__(lifetime, vertices)
		vertex_labels = np.zeros((self.lifetime, self.num_vertices()))
		self.label = label

	def add_vertex_labels(self, temporal_labels):
		self.vertex_labels = temporal_labels
		# for i, v in enumerate(self._graph.get_vertices()):
		# 	self._graph.vertex_properties[v] = self._graph.new_vertex_property("vector<bool>", temporal_labels[i])






