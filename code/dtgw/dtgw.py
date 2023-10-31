#!/usr/bin/env python3
import itertools
import enum
import time

import numpy as np
import scipy.spatial

from .dtgw_ import dtw_wrapper, warping_update_wrapper, warping_update_wrapper2, update_matchcost_wrapper, update_matchcost_wrapper2, init_diagonal_warping_wrapper, init_diagonal_warping_wrapper2, sakoe_chiba_band_wrapper

import lapjv


class AlgoState(enum.Enum):
	MATCHCOST_UPDATED = 1
	WARPCOST_UPDATED = 2
	CONVERGED = 3
	ITERATION_LIMIT_REACHED = 4


class DtgwHeuristic:
	def __init__(self, features1, features2, metric, max_iterations, dcost):
		n1, n2 = features1.shape[1], features2.shape[1]
		# if n1 == n2 and dcost == 2:
		# 	return
		if n1 >= n2:
			self._features1 = features1
			self._features2 = features2
		else:
			self._features1 = features2
			self._features2 = features1
		
		self._t1 = self._features1.shape[0]
		self._t2 = self._features2.shape[0]

		self._metric = metric
		self._dcost = dcost

		self._warppath = None
		self._matching = []
		self._warpcost = None
		self._matchcost = None
		self._state = None
		self._cost = float("inf")
		self.iteration = 0
		self.max_iterations = max_iterations
		self._min_iterations = None
		self.window = None

	@property
	def terminated(self):
		return self._state in (AlgoState.CONVERGED, AlgoState.ITERATION_LIMIT_REACHED)
		# return self._state == AlgoState.CONVERGED or self._state == AlgoState.ITERATION_LIMIT_REACHED

	@property
	def warp_path(self):
		return self._warppath
	
	@property
	def matching(self):
		return self._matching

	@property
	def cost(self):
		return self._cost

	@property
	def state(self):
		return self._state


	def sakoe_chiba_region(self, window):
		self.window = window
		# self.region = sakoe_chiba_band(self._t1, self._t2, window)
		self.region = sakoe_chiba_band_wrapper(self._t1, self._t2, window)

		
	def initialize_with_diagonal_warping(self):
		self._matchcost, self._warppath = init_diagonal_warping_wrapper2(self._features1, self._features2, self._metric, self._dcost)
		self._state = AlgoState.MATCHCOST_UPDATED
		self._min_iterations = 0

	# def initialize_with_optimistic_warping(self):
	# 	self._warpcost = np.empty((self._t1, self._t2))
	# 	for i in range(self._t1):
	# 		for j in range(self.region[0,i], self.region[1,i]):
	# 			self._warpcost[i,j] = lapjv.lapjv(self._metric(self._features1[i,:,:], self._features2[j,:,:]))[2][0]
	# 			# self._warpcost[i,j] = lap_wrapper(self._metric(self._features1[i,:,:], self._features2[j,:,:]))[0]
	# 	# self._warpcost = init_opt_warping_wrapper(self._features1, self._features2, self.region)
	# 	self._state = AlgoState.WARPCOST_UPDATED
	# 	self._min_iterations = 2

	# def initialize_with_product_warping(self):
	# 	self._warppath, self._matchcost = init_product_warping_wrapper(self._features1, self._features2, self.region)
	# 	self._state = AlgoState.MATCHCOST_UPDATED
	# 	self._min_iterations = 0

	# def initialize_with_optimistic_matching(self):
	# 	self._matchcost = init_opt_matching_wrapper(self._features1, self._features2, self.region)
	# 	self._state = AlgoState.MATCHCOST_UPDATED
	# 	self._min_iterations = 2

	def _update_cost(self, new_cost):
		# only check after we reached a valid state
		# (initialization can set unrealistic values)
		# if self.iteration > self._min_iterations and new_cost >= self._cost:
		if new_cost >= self._cost:
			self._state = AlgoState.CONVERGED
			return
		self._cost = new_cost
		
	def _check_termination(self):
		if self.iteration >= self.max_iterations:
			self._state = AlgoState.ITERATION_LIMIT_REACHED
			if self.iteration < self._min_iterations:
				warning("Heuristic terminated early, result may be invalid")
		return self.terminated

	def step(self):
		# print("iter: %d state: %s cost: %.1f" % (self.iteration, self._state, self._cost))
		# if self._state is None:
		# 	raise ValueError("Missing initialization")
		if self._check_termination():
			return
		self.iteration += 1

		if self._state == AlgoState.WARPCOST_UPDATED:
			cost, self._warppath = dtw_wrapper(self._warpcost, self.region)
			# print("tw: %.3f" % cost)
			self._update_cost(cost)
			if self._check_termination():
				return
			self._matchcost = update_matchcost_wrapper2(self._features1, self._features2, self._warppath, self._metric, self._dcost)
			self._state = AlgoState.MATCHCOST_UPDATED

		elif self._state == AlgoState.MATCHCOST_UPDATED:
			# print("calc lap...")
			# print(self._matchcost)
			# cost, self._matching = lap_wrapper(self._matchcost)
			asdf = lapjv.lapjv(self._matchcost)
			# print(asdf)
			self._matching, _, tmp = asdf
			# self._matching, _, tmp = lapjv.lapjv(self._matchcost)
			self._update_cost(tmp[0])
			# print("lapjv: %.3f" % tmp[0])
			# print(self._matching)
			if self._check_termination():
				return
			self._warpcost = warping_update_wrapper2(self._features1, self._features2, self._matching, self.region, self._metric, self._dcost)
			# print(self._warpcost)
			# print(self._matching)
			self._state = AlgoState.WARPCOST_UPDATED


def warping_path_distance(warp_path):
	return np.max(np.abs(warp_path[:,0]-warp_path[:,1]))


def compute_dtgw_log(features1, features2, dcost=0, metric="l1", init="diagonal_warping", window=None, max_iterations=100):
	# if pay:
	# 	t1, n1, k1 = features1.shape
	# 	t2, n2, k2 = features2.shape
	# 	n = max(n1, n2)
	# 	features1 = np.hstack((features1, np.broadcast_to(0, (t1, n-n1, k1)))).astype('float32')
	# 	features2 = np.hstack((features2, np.broadcast_to(0, (t2, n-n2, k2)))).astype('float32')

	heuristic = DtgwHeuristic(features1, features2, metric, max_iterations, dcost)

	if window is not None:
		heuristic.sakoe_chiba_region(window)

	if init == "diagonal_warping":
		heuristic.initialize_with_diagonal_warping()
	elif init == "optimistic_warping":
		heuristic.initialize_with_optimistic_warping()
	elif init == "optimistic_matching":
		heuristic.initialize_with_optimistic_matching()
	elif init == "sigma*":
		heuristic.initialize_with_product_warping()
	else:
		raise ValueError("Invalid initialization")

	while not heuristic.terminated:
		heuristic.step()

	# if heuristic.warp_path is not None:
	# 	wp = warping_path_distance(heuristic.warp_path)
	# else:
	# 	wp = None
	wp = warping_path_distance(heuristic.warp_path)

	if metric == "l2":
		return np.sqrt(heuristic.cost), heuristic.iteration, wp
	return heuristic.cost, heuristic.iteration, wp



def compute_dtgw_log2(features1, features2, dcost=0, metric="l1", init="diagonal_warping", window=10, max_iterations=5):

	heuristic = DtgwHeuristic(features1, features2, metric, max_iterations, dcost)

	if window is not None:
		heuristic.sakoe_chiba_region(window)

	if init == "diagonal_warping":
		heuristic.initialize_with_diagonal_warping()
	elif init == "optimistic_warping":
		heuristic.initialize_with_optimistic_warping()
	elif init == "optimistic_matching":
		heuristic.initialize_with_optimistic_matching()
	elif init == "sigma*":
		heuristic.initialize_with_product_warping()
	else:
		raise ValueError("Invalid initialization")

	while not heuristic.terminated:
		heuristic.step()

	n = features1.shape[1]
	matched = np.sum(np.arange(n) == heuristic._matching)/n

	if metric == "l2":
		return np.sqrt(heuristic.cost), matched

	return heuristic.cost, matched




import numpy.ctypeslib as npct

from ctypes import c_int, c_double
array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

libdtgw = npct.load_library("dtgw.so", "dtgw/build")

libdtgw.dtgw.restype = c_double
libdtgw.dtgw.argtypes = [array_3d_double, array_3d_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int]


def dtgw_wrapper(f1, f2, init, window, max_iterations):
	t1, t2, n, c = f1.shape[0], f2.shape[0], f1.shape[1], f1.shape[2]
	if init == "diagonal_warping":
		i = 1
	elif init == "optimistic_warping":
		i = 2
	elif init == "optimistic_matching":
		i = 3
	elif init == "sigma*":
		i = 4
	else:
		raise ValueError("Invalid initialization")
	return libdtgw.dtgw(f1, f2, t1, t2, n, c, i, window, max_iterations)


def compute_dtgw(features1, features2, eps, metric=scipy.spatial.distance.cdist, init="diagonal_warping", window=None):
	# t1, n1, k1 = features1.shape
	# t2, n2, k2 = features2.shape
	# n = max(n1, n2)
	# features1 = np.hstack((features1, np.broadcast_to(eps, (t1, n-n1, k1))))
	# features2 = np.hstack((features2, np.broadcast_to(eps, (t2, n-n2, k2))))
	t1, n1, k = features1.shape
	t2, n2, _ = features2.shape
	if n1 > n2:
		features1 = np.hstack((features1, np.broadcast_to(eps, (t1, n2-n1, k))))
	else:
		features2 = np.hstack((features2, np.broadcast_to(eps, (t2, n1-n2, k))))

	cost = dtgw_wrapper(features1, features2, init, window, 1000)
	
	return cost

def compute_tgw(features1, features2, window=100):
	t1, n, _ = features1.shape
	t2 = features2.shape[0]
	matching = np.arange(n)
	region = sakoe_chiba_band_wrapper(t1, t2, window)
	warpcost = warping_update_wrapper(features1, features2, matching, region)
	cost, warping = dtw_wrapper(warpcost, region)
	return cost



