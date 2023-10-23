"""This module contains a selection of vertex signatures.
They are stored in the SIGNATURE_PROVIDERS dict under arbitrary string keys.
Each element of this should adhere to the interface of VertexSignatureProvider.
"""

import abc
import functools
import operator

import numpy as np
import scipy.spatial
# import graph_tool as gt
# import graph_tool.centrality
# import graph_tool.topology
# import graph_tool.clustering
# from warping_wrapper import vertex_feature_neighbors_wrapper, vertex_feature_neighbors_normed_wrapper
from .dtgw_ import subtrees_feature_wrapper, subtrees_feature_wrapper2, walks_feature_wrapper, neighbors_feature_wrapper, interaction_feature_wrapper, degree_feature_wrapper
# vertex_feature_labels_wrapper


class VertexSignatureProvider(abc.ABC):
	@property
	def eps(self):
		"""This is the signature of a nonexisting vertex."""
		return self._eps

	@property
	def metric(self):
		"""This is the function that shall be used to compute the pairwise distances
		between two signature vectors (e.g. scipy.spatial.distance.cdist)
		"""
		return self._metric
	
	@abc.abstractmethod
	def signatures(self, ltg):
		"""Computes the signature vector of the given temporal graph.

		Arguments:
		----------
		ltg : LabelledTGraph

		Returns:
		--------
		signatures : np.array of shape (t, n, k)
			Vector of vertex signatures, where t is the lifetime of the temporal graph,
			and n is the number of vertices.
			signatures[i,v,:] is the signature of vertex v in layer i
		"""
		pass

SIGNATURE_PROVIDERS = {}



class SubtreesSignatureProvider(VertexSignatureProvider):
	def __init__(self, **kwargs):
		# self._eps = subtrees_feature_wrapper(tadj, tlabels, k)
		self._eps = 0

	def signatures(self, tadj, tlabels, k):
		return subtrees_feature_wrapper(tadj, tlabels, k)


SIGNATURE_PROVIDERS["subtrees"] = SubtreesSignatureProvider


class WalksSignatureProvider(VertexSignatureProvider):
	def __init__(self, **kwargs):
		# self._eps = walks_feature_wrapper(tadj, tlabels, k)
		self._eps = 0

	def signatures(self, tadj, tlabels, k):
		return walks_feature_wrapper(tadj, tlabels, k)


SIGNATURE_PROVIDERS["walks"] = WalksSignatureProvider


class NeighborsSignatureProvider(VertexSignatureProvider):
	def __init__(self, **kwargs):
		# self._eps = walks_feature_wrapper(tadj, tlabels, k)
		self._eps = 0

	def signatures(self, tadj, tlabels, k):
		return neighbors_feature_wrapper(tadj, tlabels, k)


SIGNATURE_PROVIDERS["neighbors"] = NeighborsSignatureProvider



class SubtreesSignatureProvider2(VertexSignatureProvider):
	def __init__(self, **kwargs):
		# self._eps = subtrees_feature_wrapper(tadj, tlabels, k)
		self._eps = 0

	def signatures(self, tadj, tlabels, k):
		return subtrees_feature_wrapper2(tadj, tlabels, k)


SIGNATURE_PROVIDERS["subtrees2"] = SubtreesSignatureProvider2





class DegreeSignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		pass
		# self.name = "degree%d" % k

	def signatures(self, tadj, k):
		return degree_feature_wrapper(tadj, k)


SIGNATURE_PROVIDERS["degree"] = DegreeSignatureProvider



class InteractionSignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		pass
		# self.name = "degree%d" % k

	def signatures(self, tadj, k):
		return interaction_feature_wrapper(tadj, k)


SIGNATURE_PROVIDERS["interaction"] = InteractionSignatureProvider



# class LabelSignatureProvider(VertexSignatureProvider):
# 	"""Uses vertex degrees as signatures"""
# 	def __init__(self, **kwargs):
# 		self._eps = 0
# 		self._metric = scipy.spatial.distance.cdist

# 	def signatures(self, tadj, tlabels):
# 		# return degree_matrix(ltg.tgraph)[:,:,np.newaxis]
# 		# return label_matrix(ltg.tgraph)[:,:,np.newaxis]
# 		# return ltg.tgraph.vertex_labels[:,:,np.newaxis]
# 		# return vertex_feature_labels_wrapper(tlabels)
# 		return tlabels[:,:,np.newaxis].astype('float64')
# 		# return label_matrix(ltg.tgraph)

# SIGNATURE_PROVIDERS["labels"] = LabelSignatureProvider


# class NormedNeighborhoodSignatureProvider(VertexSignatureProvider):
# 	"""Uses vertex degrees as signatures"""
# 	def __init__(self, **kwargs):
# 		self._eps = 0
# 		self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")
		
# 	def signatures(self, tadj, tlabels):
# 		return vertex_feature_neighbors_wrapper(tadj, tlabels)


# SIGNATURE_PROVIDERS["neighbors"] = NormedNeighborhoodSignatureProvider


# class NeighborhoodTwoSignatureProvider(VertexSignatureProvider):
# 	"""Uses vertex degrees as signatures"""
# 	def __init__(self, **kwargs):
# 		self._eps = 0
# 		self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

# 	def signatures(self, tadj, tlabels):
# 		return vertex_feature_neighbors2_wrapper(tadj, tlabels)


# SIGNATURE_PROVIDERS["neighbors-two"] = NeighborhoodTwoSignatureProvider


# class RandomWalksOneSignatureProvider(VertexSignatureProvider):
# 	"""Uses vertex degrees as signatures"""
# 	def __init__(self, **kwargs):
# 		self._eps = 0
# 		self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

# 	def signatures(self, tadj, tlabels):
# 		return vertex_feature_walks1_wrapper(tadj, tlabels)


# SIGNATURE_PROVIDERS["walks-one"] = RandomWalksOneSignatureProvider


# class RandomWalksTwoSignatureProvider(VertexSignatureProvider):
# 	"""Uses vertex degrees as signatures"""
# 	def __init__(self, **kwargs):
# 		self._eps = 0
# 		self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

# 	def signatures(self, tadj, tlabels):
# 		return vertex_feature_walks2_wrapper(tadj, tlabels)


# SIGNATURE_PROVIDERS["walks-two"] = RandomWalksTwoSignatureProvider


class DegreeSignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self.name = "degree1"
		self._eps = 0
		# self._metric = scipy.spatial.distance.cdist
		self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

	def signatures(self, tadj, k):
		res = np.sum(tadj, axis=2)[:,:,np.newaxis].astype('float32')
		return res
		# return vertex_feature_degree_wrapper(tadj)

SIGNATURE_PROVIDERS["degree1"] = DegreeSignatureProvider


class Degree2SignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self.name = "degree2"

	def signatures(self, tadj):
		return vertex_feature_degree2_wrapper(tadj)

SIGNATURE_PROVIDERS["degree2"] = Degree2SignatureProvider


class Degree3SignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self.name = "degree3"

	def signatures(self, tadj):
		return vertex_feature_degree3_wrapper(tadj)

SIGNATURE_PROVIDERS["degree3"] = Degree3SignatureProvider


class Degree4SignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self.name = "degree4"

	def signatures(self, tadj):
		return vertex_feature_degree4_wrapper(tadj)

SIGNATURE_PROVIDERS["degree4"] = Degree4SignatureProvider


class ActivitySignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self._eps = 0
		self._metric = scipy.spatial.distance.cdist

	def signatures(self, tfc):
		return tfc


SIGNATURE_PROVIDERS["activity"] = ActivitySignatureProvider


class NeighborsLabeledSignatureProvider(VertexSignatureProvider):
	"""Uses vertex degrees as signatures"""
	def __init__(self, **kwargs):
		self.name = "neighbors1"
		# self._eps = 0
		# self._metric = scipy.spatial.distance.cdist
		# self._metric = functools.partial(scipy.spatial.distance.cdist, metric="cityblock")

	def signatures(self, tadj):
		return tadj.astype('float64')
		# return vertex_feature_degree_wrapper(tadj)


SIGNATURE_PROVIDERS["neighbors_labeled1"] = NeighborsLabeledSignatureProvider


class TwoNeighborsLabeledSignatureProvider(VertexSignatureProvider):
	def __init__(self, **kwargs):
		self.name = "neighbors2"

	def signatures(self, tadj):
		return vertex_feature_lneighbors2_wrapper(tadj)


SIGNATURE_PROVIDERS["neighbors_labeled2"] = TwoNeighborsLabeledSignatureProvider



class ThreeNeighborsLabeledSignatureProvider(VertexSignatureProvider):
	def __init__(self, **kwargs):
		self.name = "neighbors3"

	def signatures(self, tadj):
		return vertex_feature_lneighbors3_wrapper(tadj)


SIGNATURE_PROVIDERS["neighbors_labeled3"] = ThreeNeighborsLabeledSignatureProvider


