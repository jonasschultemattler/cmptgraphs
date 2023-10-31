import abc
import os
import numpy as np
from tqdm import tqdm
import networkx as nx
import igraph as ig
# import grakel as gk
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




class TemporalGraph():
	def __init__(self, tadj):
		self.tadj = tadj

	def save(self, path):
		pass



class LabeledTemporalGraph(TemporalGraph):
	def __init__(self, tadj, tlabels):
		super().__init__(tadj)
		self.tlabels = tlabels



class BrainDataloader(abc.ABC):
	def __init__(self, name, dir_path):
		self.name = name
		self.dir_path = dir_path
		self.path = os.path.join(dir_path, self.name)
		self.n = 0
		self.data = None
		self.roi = None
		self.roi_name = ""
		self.masker = None


	@abc.abstractmethod
	def load_data(self, n_subjects):
		pass

	@abc.abstractmethod
	def get_labels(self):
		pass

	@abc.abstractmethod
	def subjects_loaded(self, path):
		pass

	@abc.abstractmethod
	def compute_timeseries(self):
		pass


	def get_data(self):
		tadjs = self.get_temporal_graphs()
		labels = self.get_labels()
		# return self.dict_to_numpy(tadjs, labels)
		return self.dict_to_list(tadjs, labels)

	def get_nx_data(self):
		tgraphs = self.temporal_graphs_to_nx()
		labels = self.get_labels()
		return self.dict_to_list(tgraphs, labels)

	def get_ig_data(self):
		tgraphs = self.temporal_graphs_to_igraphs()
		labels = self.get_labels()
		return self.dict_to_list(tgraphs, labels)

	# def get_gk_data(self):
	# 	tgraphs = self.temporal_graphs_to_grakel()
	# 	labels = self.get_labels()
	# 	return self.dict_to_list(tgraphs, labels)


	def get_timeseries_data(self):
		timeseries = self.get_timeseries()
		labels = self.get_labels()
		# return self.dict_to_numpy(timeseries, labels)
		return self.dict_to_list(timeseries, labels)


	def dict_to_numpy(self, data_dict, labels_dict):
		first = True
		for key, data in data_dict.items():
			if first:
				X = data[np.newaxis,:]
				y = labels_dict[key]
				first = False
			else:
				X = np.vstack((X, data[np.newaxis,:]))
				y = np.hstack((y, labels_dict[key]))
		return X, y


	def dict_to_list(self, data_dict, labels_dict):
		data_list, label_list = [], []
		for key, data in data_dict.items():
			data_list.append(data)
			label_list.append(labels_dict[key])
		return data_list, label_list


	def load_roi(self, roi):
		if roi == "atlas":
			self.roi = datasets.fetch_atlas_msdl(data_dir=os.path.join(self.dir_path, 'roi'))
		# if roi == "schaefer":
		# 	self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(self.dir_path, 'roi'))
		# if roi == "aal":
		# 	self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(path, 'roi'))
		# if roi == "destrieux":
		# 	self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(path, 'roi'))
		# if roi == "harvard_oxford":
		# 	self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(path, 'roi'))

	# def get_masker(self):
	# 	# if roi == "schaefer":
	# 	# 	self.masker = NiftiLabelsMasker(self.roi["maps"])
	# 	# if roi == "atlas":
	# 	if self.roi is None:
	# 		self.load_roi(self.roi_name)
	# 	self.masker = NiftiMapsMasker(self.roi.maps, resampling_target="data",
	# 		t_r=2, detrend=True, low_pass=0.1, high_pass=0.01, standardize="zscore_sample").fit()


	def get_timeseries(self):
		path = os.path.join(self.path, "timeseries")
		if os.path.exists(path):
			if not self.subjects_loaded(path):
				print("not all timeseries computed.")
				time_series = self.compute_timeseries()
			else:
				print("loading timeseries...")
				time_series = {s: np.load(os.path.join(path, s + "_" + self.roi_name + ".npy")) for s in self.subjects}
		else:
			os.mkdir(path)
			time_series = self.compute_timeseries()
		return time_series


	def get_temporal_graphs(self):
		if os.path.exists(os.path.join(self.path, "temporal_graphs")):
			if not self.subjects_loaded(os.path.join(self.path, "temporal_graphs")):
				print("not all temporal_graphs computed.")
				return self.build_temporal_graphs()
			else:
				print("loading temporal_graphs...")
				tadjs = {subject[:12]: np.load(os.path.join(self.path, "temporal_graphs", subject + "_" + self.roi_name + ".npy")) for subject in self.subjects}
				return tadjs
		else:
			return self.build_temporal_graphs()


	def build_temporal_graphs(self):
		timeseries = self.get_timeseries()
		if not os.path.exists(os.path.join(self.path, "temporal_graphs")):
			os.mkdir(os.path.join(self.path, "temporal_graphs"))
		tadjs = {}
		print("building temporal_graphs...")
		for name, time_series in tqdm(timeseries.items()):
			tadj = self.build_temporal_graph(time_series)
			np.save(os.path.join(self.path, "temporal_graphs", name + "_" + self.roi_name), tadj)
			tadjs.update({name: tadj})
		return tadjs


	def build_temporal_graph(self, time_series, window=50, stride=3, percentile=30):
		# points = np.arange(0, time_series.shape[0]-window, stride)
		# T, n = len(points), time_series.shape[1]
		fc = self.functional_connectivity(time_series, window, stride)
		tadj = np.empty(fc.shape, dtype=bool)
		for t in range(len(fc)):
		# for t, p in enumerate(points):
			# fc = np.corrcoef(time_series[p:p+window,:].T)
			fc_uh = fc[t][np.triu_indices(fc[t].shape[0], k=1)]
			p = np.percentile(fc_uh, percentile)
			adj = fc[t] > p
			# adj = fc[t] > np.percentile(fc[t], percentile)
			tadj[t,:,:] = adj & (~np.eye(fc[t].shape[0], dtype=bool))
		return tadj


	def functional_connectivity(self, time_series, window, stride):
	    points = np.arange(0, time_series.shape[0]-window, stride)
	    T, n = len(points), time_series.shape[1]
	    # fc = np.empty((T, (n*n - n)//2))
	    fc = np.empty((T, n, n))
	    for t, p in enumerate(points):
	        corr = np.corrcoef(time_series[p:p+window,:].T)
	        # corr = np.correlate(time_series[p:p+window,:].T)
	        # fc[t] = corr[np.triu_indices(n, k=1)]
	        fc[t] = corr
	    return fc


	def savetxt(self, distinct_labels):
		path = os.path.join(self.path, "temporal_graphs")
		X, y = self.get_data()
		edges_file = open(os.path.join(path, self.name + "_A.txt"), 'w')
		tedges_file = open(os.path.join(path, self.name + "_edge_attributes.txt"), 'w')
		c = 0
		for tadj in X:
			T, n, _ = tadj.shape
			for i in range(n):
				for t in range(T):
					for j in range(n):
						if tadj[t][i][j]:
							edges_file.write("%d, %d\n" % (c+i+1, c+j+1))
							tedges_file.write("%d\n" % (t+1))
			c += n
		graph_file = open(os.path.join(path, self.name + "_graph_indicator.txt"), 'w')
		for g, tadj in enumerate(X):
			for i in range(tadj.shape[1]):
				graph_file.write("%d\n" % (g+1))
		labels_file = open(os.path.join(path, self.name + "_graph_labels.txt"), 'w')
		for label in y:
			labels_file.write("%d\n" % label)
		nodes_label_file = open(os.path.join(path, self.name + "_node_labels.txt"), 'w')
		if distinct_labels:
			for tadj in X:
				for i in range(tadj.shape[1]):
					nodes_label_file.write("0, %d\n" % i)
		else:
			for tadj in X:
				for i in range(tadj.shape[1]):
					nodes_label_file.write("0, 0\n")


	def temporal_graphs_to_nx(self):
		tgraphs = self.get_temporal_graphs()
		Gs = {}
		for name, tadj in tgraphs.items():
			G = {}
			for t, adj in enumerate(tadj):
				G[t] = nx.from_numpy_array(adj)
			Gs[name] = G
		return Gs


	def temporal_graphs_to_igraphs(self):
		tgraphs = self.get_temporal_graphs()
		TGs = {}
		for name, tadj in tgraphs.items():
			Gs = [ig.Graph.Adjacency(adj) for adj in tadj]
			_, n, _ = tadj.shape
			v_labels = [i for i in range(n)]
			for g in Gs:
				# g.vs.['label'] = v_labels
				g.vs.set_attribute_values('label', v_labels)
			TGs[name] = Gs
		return TGs


	# def temporal_graphs_to_grakel(self):
	# 	tgraphs = self.get_temporal_graphs()
	# 	TGs = {}
	# 	for name, tadj in tgraphs.items():
	# 		_, n, _ = tadj.shape
	# 		# v_labels = [i for i in range(n)]
	# 		v_labels = {i:i for i in range(n)}
	# 		Gs = [gk.Graph(adj, node_labels=v_labels) for adj in tadj]
	# 		# Gs = [gk.Graph(adj) for adj in tadj]
	# 		TGs[name] = Gs
	# 	return TGs



class BrainDevelopementDataloader(BrainDataloader):
	def __init__(self, name, path, n, roi_name):
		super().__init__(name, path)
		self.n = min(n, 100)
		self.roi_name = roi_name
		self.load_data()
		self.subjects = [os.path.split(self.data.func[i])[1][:12] for i in range(len(self.data.func))]


	def load_data(self):
		print("loading fmri data...")
		self.data = datasets.fetch_development_fmri(n_subjects=self.n, data_dir=self.path)


	def get_masker(self):
		if self.roi is None:
			self.load_roi(self.roi_name)
		self.masker = NiftiMapsMasker(self.roi.maps, resampling_target="data",
			t_r=2, detrend=True, low_pass=0.1, high_pass=0.01, standardize="zscore_sample").fit()


	def subjects_loaded(self, path):
		subjects = [f[:12] for f in os.listdir(path) if f.endswith('.npy')]
		for s in self.subjects:
			if s not in subjects:
				return False
		return len(subjects) >= self.n


	def compute_timeseries(self):
		if self.masker is None:
			self.get_masker()
		time_series = {}
		print("compute timeseries...")
		for i in tqdm(range(len(self.data.func))):
			timeseries = self.masker.fit_transform(self.data.func[i], confounds=self.data.confounds[i])
			_, subject = os.path.split(self.data.func[i])
			time_series.update({subject[:12]: timeseries})
			np.save(os.path.join(os.path.join(self.path, "timeseries"), subject[:12] + "_" + self.roi_name), timeseries)
		return time_series


	def get_labels(self):
		label_key = {"child": 0, "adult": 1}
		labels_dict = {}
		for i in range(self.data.phenotypic.shape[0]):
			labels_dict.update({self.data.phenotypic[i][0]: label_key[self.data.phenotypic[i][3]]})
		return labels_dict




class AbideDataloader(BrainDataloader):
	def __init__(self, name, path, n, roi_name):
		super().__init__(name, path)
		self.n = min(n, 100)
		self.roi_name = roi_name
		self.load_data()
		self.subjects = [os.path.split(self.data.func_preproc[i])[1][:-20] for i in range(len(self.data.func_preproc))]


	def load_data(self):
		print("loading fmri data...")
		self.data = datasets.fetch_abide_pcp(n_subjects=self.n, data_dir=self.path)


	def get_masker(self):
		if self.roi is None:
			self.load_roi(self.roi_name)
		self.masker = NiftiMapsMasker(self.roi.maps, resampling_target="data",
			t_r=2, detrend=True, low_pass=0.1, high_pass=0.01, standardize="zscore_sample").fit()
		# power = datasets.fetch_coords_power_2011()
		# coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T
		# self.masker = input_data.NiftiSpheresMasker(seeds=coords, smoothing_fwhm=4, radius=5., standardize=True, detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5)
		# self.masker = NiftiMapsMasker(self.roi.maps, detrend=True, standardize=True).fit()


	def subjects_loaded(self, path):
		subjects = [f[:-10] for f in os.listdir(path) if f.endswith('.npy')]
		for s in self.subjects:
			if s not in subjects:
				return False
		return len(subjects) >= self.n


	def compute_timeseries(self):
		if self.masker is None:
			self.get_masker()
		time_series = {}
		print("compute timeseries...")
		for i in tqdm(range(len(self.data.func_preproc))):
			timeseries = self.masker.fit_transform(self.data.func_preproc[i])
			_, subject = os.path.split(self.data.func_preproc[i])
			time_series.update({subject[:-20]: timeseries})
			np.save(os.path.join(os.path.join(self.path, "timeseries"), subject[:-20] + "_" + self.roi_name), timeseries)
		return time_series


	def get_labels(self):
		labels_dict = {}
		for i in range(self.data.phenotypic.shape[0]):
			labels_dict.update({self.data.phenotypic[i][6]: self.data.phenotypic[i][7]-1})
		return labels_dict



class ADHDDataloader(BrainDataloader):
	def __init__(self, name, path, n, roi_name):
		super().__init__(name, path)
		print(self.path)
		self.n = min(n, 40)
		self.roi_name = roi_name
		self.load_data()
		self.subjects = [os.path.split(self.data.func[i])[1][:7] for i in range(len(self.data.func))]


	def load_data(self):
		print("loading fmri data...")
		self.data = datasets.fetch_adhd(n_subjects=self.n, data_dir=self.path)


	def get_masker(self):
		if self.roi is None:
			self.load_roi(self.roi_name)
		self.masker = NiftiMapsMasker(self.roi.maps, resampling_target="data",
			t_r=2, detrend=True, low_pass=0.1, high_pass=0.01, standardize="zscore_sample").fit()


	def subjects_loaded(self, path):
		subjects = [f[:7] for f in os.listdir(path) if f.endswith('.npy')]
		for s in self.subjects:
			if s not in subjects:
				return False
		return len(subjects) >= self.n


	def compute_timeseries(self):
		if self.masker is None:
			self.get_masker()
		time_series = {}
		print("compute timeseries...")
		for i in tqdm(range(len(self.data.func))):
			timeseries = self.masker.fit_transform(self.data.func[i], confounds=self.data.confounds[i])
			_, subject = os.path.split(self.data.func[i])
			print(subject)
			time_series.update({subject[:7]: timeseries})
			np.save(os.path.join(os.path.join(self.path, "timeseries"), subject[:7] + "_" + self.roi_name), timeseries)
		return time_series


	def get_labels(self):
		labels_dict = {}
		for i in range(self.data.phenotypic.shape[0]):
			labels_dict.update({str(self.data.phenotypic[i][1]).zfill(7): self.data.phenotypic[i][22]})
		return labels_dict



