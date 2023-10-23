from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


path = "../datasets/brains/development"
n_subjects = 30

development_dataset = datasets.fetch_development_fmri(n_subjects=n_subjects, data_dir=path)
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords

masker = NiftiMapsMasker(
    msdl_data.maps,
    resampling_target="data",
    t_r=2,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    memory="nilearn_cache",
    memory_level=1,
    standardize="zscore_sample",
).fit()

children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
    development_dataset.func,
    development_dataset.confounds,
    development_dataset.phenotypic,
):
    time_series = masker.transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    if phenotypic["Child_Adult"] == "child":
        children.append(time_series)
    groups.append(phenotypic["Child_Adult"])


children_percentage = len(children)/len(pooled_subjects)
print(f"Data has {children_percentage*100}% children.")


kinds = ["correlation", "partial correlation", "tangent"]
_, classes = np.unique(groups, return_inverse=True)
print(classes.shape)
# cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
cv = StratifiedShuffleSplit()
pooled_subjects = np.asarray(pooled_subjects)
print(pooled_subjects.shape)

scores = {}
for kind in kinds:
    scores[kind] = []
    for train, test in cv.split(pooled_subjects, classes):
        # *ConnectivityMeasure* can output the estimated subjects coefficients
        # as a 1D arrays through the parameter *vectorize*.
        connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
        # build vectorized connectomes for subjects in the train set
        connectomes = connectivity.fit_transform(pooled_subjects[train])
        # fit the classifier
        classifier = LinearSVC().fit(connectomes, classes[train])
        # make predictions for the left-out test subjects
        predictions = classifier.predict(
            connectivity.transform(pooled_subjects[test])
        )
        # store the accuracy for this cross-validation fold
        scores[kind].append(accuracy_score(classes[test], predictions))


mean_scores = [np.mean(scores[kind]) for kind in kinds]
scores_std = [np.std(scores[kind]) for kind in kinds]

plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * 0.1 + 0.1
plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
yticks = [k.replace(" ", "\n") for k in kinds]
plt.yticks(positions, yticks)
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.gca().axvline(children_percentage, color="red", linestyle="--")
plt.xlabel("Classification accuracy\n(red line = chance level)")
plt.tight_layout()

plt.show()


