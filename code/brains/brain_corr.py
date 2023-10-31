from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score, pairwise
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.svm import LinearSVC
from sklearn import svm
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

import scipy.spatial

from brain_dataloader2 import BrainDevelopementDataloader
# from compute_brain_distances import distance_matrix_tw
# from warping_wrapper import dtw_wrapper, sakoe_chiba_band_wrapper


def norm_matrix(matrix):
    nmatrix = np.empty(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            d = np.sqrt(matrix[i, i]) * np.sqrt(matrix[j, j]);
            if d != 0:
                nmatrix[i, j] = matrix[i, j]/d;
            else:
                nmatrix[i, j] = 0;
    return nmatrix


def corr_kernel(data, kind="correlation"):
    connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
    connectomes = connectivity.fit_transform(data)
    kernel = pairwise.linear_kernel(connectomes)
    return kernel


path = "../../datasets/brains"
dataset = "development"
roi = "atlas"
n_subjects = 100
output_path = "../../output"


dataloader = BrainDevelopementDataloader(dataset, path, n_subjects, roi)
timeseries, labels = dataloader.get_timeseries_data()

kernel = corr_kernel(timeseries)
kernel = norm_matrix(kernel)

print(np.min(kernel))
print(np.max(kernel))
print(kernel.shape)
print(kernel)

np.savetxt(os.path.join(output_path, "brains", dataset, "corr.gram"), kernel)




