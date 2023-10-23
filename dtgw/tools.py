import itertools
import collections
import time

import numpy as np

from config import CONFIG




class OutputSink:
	def __init__(self, file, flags=set()):
		self._file = file
		self.flags = flags

	def write(self, str):
		self._file.write(str)


class LoggingObserver:
	def __init__(self):
		"""Logging dtgw observer

		Arguments:
		----------
		outfile :
			File to write to
		vertex_map_1, vertex_map_2:
			Functions to apply to any vertex from the first resp. second
			temporal graph before outputting
		layer_map_1, layer_map_2:
			Functions to apply to any layer of the first resp. second
			temporal graph before outputting
		final_only:
			Whether to log only the final result (True) or intermediate
			states, too (False)
		"""
		self._sinks = []
		self._vertex_map_1 = str
		self._vertex_map_2 = str
		self._layer_map_1 = str
		self._layer_map_2 = str
		self._features1 = None
		self._features2 = None
		self._metric = None
		self._start = time.time()

	def set_output_maps(self, vertex_map_1, vertex_map_2, layer_map_1, layer_map_2):
		self._vertex_map_1 = vertex_map_1
		self._vertex_map_2 = vertex_map_2
		self._layer_map_1 = layer_map_1
		self._layer_map_2 = layer_map_2

	def add_sink(self, sink):
		self._sinks.append(sink)

	def _log(self, msg, flags):
		for sink in self._sinks:
			if flags <= sink.flags:
				sink.write(msg + "\n")

	def _has_sink(self, flags):
		for sink in self._sinks:
			if flags <= sink.flags:
				return True
		return False

	def _time(self):
		return time.time() - self._start

	def update_features(self, features1, features2, metric):
		self._log("Features computed (time = {:.3f}s)".format(self._time()), {"progress"})
		self._features1 = features1
		self._features2 = features2
		self._metric = metric
		if self._has_sink({"features"}):
			self._log("TGraph1 features:\n" + str(self._features1) + "\n", {"features"})
			self._log("TGraph2 features:\n" + str(self._features2) + "\n", {"features"})

	def _warp_cost(self, vertex_matching, i, j):
		cost = self._metric(self._features1[i,:,:], self._features2[j,:,:])
		return sum(cost[v,w] for (v, w) in vertex_matching)

	def _match_cost(self, warp_path, v, w):
		cost = self._metric(self._features1[:,v,:], self._features2[:,w,:])
		return sum(cost[i,j] for (i, j) in warp_path)

	def update_state(self, state, matching, warp_path, cost, iteration=None, algo_state=None):
		self._log("\n=========================\n", {"progress"})
			
		flags = {"result"} if state == "done" else {"progress"}
		self._log("Time = {:.3f}s".format(self._time()), flags)
		if iteration is not None:
			self._log("Iteration = {}".format(iteration), flags)
		if algo_state is not None:
			self._log("State = {}".format(algo_state), flags)
		self._log("Cost = {}".format(cost), flags)

		flags = {"result"} if state == "done" else {"full progress"}
		if self._has_sink(flags):
			# we may need to iterate those multiple times:
			matching = ensure_container(matching)
			warp_path = ensure_container(warp_path)

			if state == "done" or state == "heuristic intialized":
				self._log_warp_path(warp_path, matching, flags)
				self._log("", flags)
				self._log_matching(matching, warp_path, flags)
			elif state == "matching updated":
				self._log_matching(matching, warp_path, flags)
			elif state == "warp path updated":
				self._log_warp_path(warp_path, matching, flags)

	def update_progress(self, msg):
		self._log(msg + " (time = {:.3f}s)".format(self._time()), {"progress"})


	def _log_matching(self, matching, warp_path, flags):
		for sink in self._sinks:
			if flags <= sink.flags:
				if "detailed cost" in sink.flags:
					sink.write(print_vertex_matching_with_cost(
						matching,
						lambda v, w: self._match_cost(warp_path, v, w),
						self._vertex_map_1,
						self._vertex_map_2
					))
				else:
					sink.write(print_vertex_matching(
						matching,
						self._vertex_map_1,
						self._vertex_map_2
					))

	def _log_warp_path(self, warp_path, matching, flags):
		for sink in self._sinks:
			if flags <= sink.flags:
				if "detailed cost" in sink.flags:
					sink.write(print_warping_with_cost(
						warp_path,
						lambda i, j: self._warp_cost(matching, i, j),
						self._layer_map_1,
						self._layer_map_2
					))
				else:
					sink.write(print_warping(
						warp_path,
						self._layer_map_1,
						self._layer_map_2,
						abbrev=True
					))


def info(msg):
	if CONFIG["verbosity"] >= 1:
		print(msg)

def warning(msg):
	if CONFIG["verbosity"] >= 1:
		print("WARNING: ", msg)


def print_vertex_matching(matching, vertex_map_1=str, vertex_map_2=str):
	"""Print vertex matching as a human-readable string."""
	output = ""
	for v, w in matching:
		output += str(vertex_map_1(v)) + " <=> " + str(vertex_map_2(w)) + "\n"
	return output


def print_vertex_matching_with_cost(matching, cost_fn, vertex_map_1=str, vertex_map_2=str):
	"""Print vertex matching as a human-readable string."""
	output = ""
	for v, w in matching:
		output += str(vertex_map_1(v)) + " <=> " + str(vertex_map_2(w))
		output += " cost=" + str(cost_fn(v, w)) + "\n"
	return output


def last(iterable):
	return collections.deque(iterable, maxlen=1).pop()


def split_before(predicate, iterable):
	"""Splits an iterable at the first occurence of an element satisfying the predicate.
	Returns two iterators. The first produces all elements up to the first element satisfying the predicate.
	The second produces all elements starting at that one.

	Example
	--------
	(a, b) = split_before(lambda x: x == 0, range(-3, 3))
	list(b) # [0, 1, 2]
	list(a) # [-3, -2, -1]
	"""
	iterable = iter(iterable)
	found = False
	buffer = []
	def first_part():
		nonlocal iterable, found
		while not found:
			try:
				item = next(iterable)
			except StopIteration:
				return
			if predicate(item):
				found = True
				iterable = itertools.chain([item], iterable)
			else:
				yield item
		yield from buffer
	def second_part():
		nonlocal found
		while not found:
			try:
				item = next(iterable)
			except StopIteration:
				return
			if predicate(item):
				found = True
				yield item
			else:
				buffer.append(item)
		yield from iterable
	return first_part(), second_part()


def print_warping(warp_path, layer_map_1=str, layer_map_2=str, abbrev=False):
	"""Print warping path as a human readable string.
	If »abbrev« is True, use an abbreviated notation.
	"""
	if abbrev:
		return print_warping_abbrev(warp_path, layer_map_1, layer_map_2)
	output = ""
	for a, b in warp_path:
		output += str(layer_map_1(a)) + " --- " + str(layer_map_2(b)) + "\n"
	return output


def print_warping_abbrev(warp_path, layer_map_1=str, layer_map_2=str):
	output = ""
	it = iter(warp_path)
	try:
		while True:
			a, b = next(it)
			try:
				a2, b2 = next(it)
				if a == a2:
					# left star
					chunk, it = split_before(lambda x: x[0] != a, it)
				elif b == b2:
					# right star
					chunk, it = split_before(lambda x: x[1] != b, it)
				else:
					# trivial (diagonal) warping
					chunk, it = split_before(lambda x: x[1]-x[0] != b-a, it)
				try:
					a2, b2 = last(chunk)
				except IndexError:
					pass # chunk is empty
			except StopIteration:
				a2, b2 = a, b # reached the end of warp_path
			output += str(layer_map_1(a))
			if a2 != a:
				output += ".." + str(layer_map_1(a2))
			output += " --- " + str(layer_map_2(b))
			if b2 != b:
				output += ".." + str(layer_map_2(b2))
			output += "\n"
	except StopIteration:
		return output


def print_warping_with_cost(warp_path, cost_fn, layer_map_1=str, layer_map_2=str):
	"""Print warping path as a human readable string.
	"""
	output = ""
	for i, j in warp_path:
		output += str(layer_map_1(i)) + " --- " + str(layer_map_2(j))
		output += " cost=" + str(cost_fn(i, j)) + "\n"
	return output


def shortest_warp_path(n, m):
	i, j = 0, 0
	while i < n and j < m:
		yield i, j
		a = (i+1) * m
		b = (j+1) * n
		if a < b + n:
			i += 1
		if b < a + m:
			j += 1


def shortest_warp_path2(n, m):
	path = np.zeros((n+m,2), dtype=int)
	# path = np.empty((n+m,2), dtype=int)
	# path[0,0] = path[0,1] = 0
	i, j = 0, 0
	l = 1
	while i < n and j < m:
		a = (i+1) * m
		b = (j+1) * n
		if a < b + n:
			i += 1
		if b < a + m:
			j += 1
		path[l][0] = i
		path[l][1] = j
		l += 1
	return path[:l-1,:]


def ensure_container(iterable):
	"""If iterable is itself an iterator, convert it to a list.
	Otherwise return it unchanged.
	"""
	if iter(iterable) is iterable:
		return list(iterable)
	return iterable
