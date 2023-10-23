from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.svm import LinearSVC
from sklearn import svm
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

import scipy.spatial

from brain_dataloader2 import BrainDevelopementDataloader, AbideDataloader
# from compute_brain_distances import distance_matrix_tw
# from warping_wrapper import dtw_wrapper, sakoe_chiba_band_wrapper

path = "../../datasets/brains"
dataset = "abide"
roi = "atlas"
n_subjects = 100
output_path = "../../output"


dataloader = AbideDataloader(path, dataset, n_subjects, roi)
# timeseries, labels = dataloader.get_timeseries_data()
# dataloader.savetxt()

# distances = distance_matrix_tw(timeseries, labels, output_path, dataset, 10, scipy.spatial.distance.cdist)


# children_percentage = (labels.shape[0] - np.sum(labels))/labels.shape[0]


# kinds = ["correlation", "partial correlation", "tangent"]
# classes = labels
# cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
# cv = StratifiedShuffleSplit()
# pooled_subjects = timeseries

# scores = {}
# for kind in kinds:
#     scores[kind] = []
#     for train, test in cv.split(pooled_subjects, classes):
#         # *ConnectivityMeasure* can output the estimated subjects coefficients
#         # as a 1D arrays through the parameter *vectorize*.
#         connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
#         # build vectorized connectomes for subjects in the train set
#         connectomes = connectivity.fit_transform(pooled_subjects[train])
#         # fit the classifier
#         classifier = LinearSVC().fit(connectomes, classes[train])
#         # make predictions for the left-out test subjects
#         predictions = classifier.predict(
#             connectivity.transform(pooled_subjects[test])
#         )
#         # store the accuracy for this cross-validation fold
#         scores[kind].append(accuracy_score(classes[test], predictions))


# mean_scores = [np.mean(scores[kind]) for kind in kinds]
# scores_std = [np.std(scores[kind]) for kind in kinds]


# plt.figure(figsize=(6, 4))
# positions = np.arange(len(kinds)) * 0.1 + 0.1
# plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
# yticks = [k.replace(" ", "\n") for k in kinds]
# plt.yticks(positions, yticks)
# plt.gca().grid(True)
# plt.gca().set_axisbelow(True)
# plt.gca().axvline(children_percentage, color="red", linestyle="--")
# plt.xlabel("Classification accuracy\n(red line = chance level)")
# plt.tight_layout()


# scores = {}
# for kind in kinds:
#     scores[kind] = []
#     for train, test in cv.split(pooled_subjects, classes):
#         connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
#         connectomes = connectivity.fit_transform(pooled_subjects[train])
#         classifier = svm.SVC(kernel='linear').fit(connectomes, classes[train])
#         connectomes = connectivity.transform(pooled_subjects[test])
#         predictions = classifier.predict(connectomes)
#         scores[kind].append(accuracy_score(classes[test], predictions))

# mean_scores = [np.mean(scores[kind]) for kind in kinds]
# scores_std = [np.std(scores[kind]) for kind in kinds]


# plt.figure(figsize=(6, 4))
# positions = np.arange(len(kinds)) * 0.1 + 0.1
# plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
# yticks = [k.replace(" ", "\n") for k in kinds]
# plt.yticks(positions, yticks)
# plt.gca().grid(True)
# plt.gca().set_axisbelow(True)
# plt.gca().axvline(children_percentage, color="red", linestyle="--")
# plt.xlabel("Classification accuracy\n(red line = chance level)")
# plt.tight_layout()

# if not os.path.exists(output_path):
#     os.mkdir(output_path)
# output_path = os.path.join(output_path, "brains")
# if not os.path.exists(output_path):
#     os.mkdir(output_path)
# output_path = os.path.join(output_path, dataset)
# if not os.path.exists(output_path):
#     os.mkdir(output_path)

# plt.savefig(os.path.join(output_path, "svc4.png"))


def functional_connectivity(time_series, window, stride):
    points = np.arange(0, time_series.shape[0]-window, stride)
    T, n = len(points), time_series.shape[1]
    fc = np.empty((T, (n*n - n)//2))
    for t, p in enumerate(points):
        corr = np.corrcoef(time_series[p:p+window,:].T)
        # corr = np.correlate(time_series[p:p+window,:].T)
        fc[t] = corr[np.triu_indices(n, k=1)]
    return fc


def fcs_similarity(fcsi, fcsj):
    similarity_matrix = np.ones((len(fcsi), len(fcsj)))
    for i, fci in enumerate(fcsi):
        for j, fcj in enumerate(fcsj):
            if j > i:
                d = np.dot(fci.flatten().T, fcj.flatten())
                similarity_matrix[i,j] = similarity_matrix[j,i] = d
    return similarity_matrix


def tw_matrix(fcs):
    matrix = np.empty((len(fcs), len(fcs)))
    for i, fci in enumerate(fcs):
        for j, fcj in enumerate(fcs):
            if j >= i:
                cost_matrix = -fcs_similarity(fci, fcj)
                region = sakoe_chiba_band_wrapper(fci.shape[0], fcj.shape[0], 10)
                res, _ = dtw_wrapper(cost_matrix, region)
                matrix[i,j] = matrix[j,i] = -res
    return matrix


def train_svm(timeseries, labels, window, stride):
    fcs = np.array([functional_connectivity(ts, window, stride).flatten() for ts in timeseries])
    scores = []
    cv = StratifiedShuffleSplit(random_state=0)
    for train, test in cv.split(timeseries, labels):
        svc = svm.SVC(kernel='linear').fit(fcs[train], labels[train])
        predictions = svc.predict(fcs[test])
        scores.append(accuracy_score(labels[test], predictions))
    return np.mean(scores), np.std(scores)


def train_svm2(timeseries, labels, window, stride):
    fcs = [functional_connectivity(ts, window, stride) for ts in timeseries]
    print(len(fcs))
    kernel = tw_matrix(fcs)
    print(kernel)
    # kernel /= np.max(kernel)
    scores = []
    cv = StratifiedShuffleSplit(random_state=0)
    for train, test in cv.split(timeseries, labels):
        svc = svm.SVC(kernel='precomputed').fit(kernel[train][:,train], labels[train])
        predictions = svc.predict(kernel[test][:,train])
        scores.append(accuracy_score(labels[test], predictions))
    return np.mean(scores), np.std(scores)



lifetime = timeseries[0].shape[0]
print(lifetime)

windows = np.arange(10, lifetime-2, 20)
strides = 2**np.arange(2)
# windows = np.array([lifetime-2])
# strides = np.array([1])

accuracies = np.zeros((len(windows), len(strides)))
for i, window in tqdm(enumerate(windows)):
    print(window)
    for j, stride in enumerate(strides):
        print(stride)
        accuracies[i,j] = train_svm2(timeseries, labels, window, stride)[0]
        # accuracies[i,j] = train_svm(timeseries, labels, window, stride)[0]
# np.save(os.path.join(output_path, "accuracies.npy"), accuracies)

# accuracies = np.load(os.path.join(output_path, "accuracies.npy"))

fig, axes = plt.subplots(1, figsize=(16,9))
im = axes.imshow(accuracies.T)
axes.set_xticks(np.arange(len(windows)), labels=windows)
axes.set_yticks(np.arange(len(strides)), labels=strides)
axes.set_xlabel("window size")
axes.set_ylabel("stride")
for i in range(len(windows)):
    for j in range(len(strides)):
        axes.text(i, j, "%.3f" % accuracies[i, j], ha="center", va="center", color="w")
axes.figure.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(output_path, "svc5.png"))


