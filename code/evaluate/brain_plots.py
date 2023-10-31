import os
from nilearn import datasets
from nilearn.maskers import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import brains.brain_dataloader as dataloader


dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
    "Posterior Cingulate Cortex",
    "Left Temporoparietal junction",
    "Right Temporoparietal junction",
    "Medial prefrontal cortex",
]

# Loading the functional datasets

dataset_path = os.path.join("..", "datasets", "brains")

dl = dataloader.BrainDevelopementDataloader("development", dataset_path, 1, "atlas")

func_filename = dl.data.func[0]
confounds_filename = dl.data.confounds[0]

figure_path = os.path.join("..", "figures")


img4d = image.load_img(func_filename)
first_img = image.index_img(img4d, 10)
print(first_img.shape)
# plotting.plot_stat_map(func_filename, output_file=os.path.join(figure_path, "test.pdf"))
plotting.plot_glass_brain(first_img, display_mode="x", output_file=os.path.join(figure_path, "test.pdf"))


masker = NiftiSpheresMasker(
    dmn_coords,
    radius=8,
    detrend=True,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    low_pass=0.1,
    high_pass=0.01,
    t_r=2,
    memory="nilearn_cache",
    memory_level=1,
    verbose=2,
    clean__butterworth__padtype="even",
)


time_series = masker.fit_transform(func_filename, confounds=[confounds_filename])


fig, ax = plt.subplots()

timeseries = time_series.T[:,:10]
ax.imshow(timeseries, cmap='hot')
ax.plot([1.5,6.5,6.5,1.5,1.5], [3.5,3.5,-0.5,-0.5,3.5], color="black", linewidth=5)
# ax.arrow(5.5, 1.5, 2, 0, width=0.1, color="black")


ax.set_yticks([])
ax.set_xticks([])
fig.tight_layout()

fig.savefig(os.path.join(figure_path, "timeseries.pdf"))



fcs = dl.functional_connectivity(timeseries.T, 5, 2)

fig, ax = plt.subplots(1, len(fcs))

for i, fc in enumerate(fcs):
	ax[i].imshow(fc, cmap='hot')
	ax[i].set_xticks([])
	ax[i].set_yticks([])

fig.tight_layout()
fig.savefig(os.path.join(figure_path, "fcs.pdf"))




fig, ax = plt.subplots(1, len(fcs))

for i, fc in enumerate(fcs):
	fc_uh = fc[np.triu_indices(fc.shape[0], k=1)]
	p30 = np.percentile(fc_uh, 30)
	adj = fc > p30
	adj = adj & (~np.eye(fc.shape[0], dtype=bool))
	plotting.plot_connectome(adj.astype('float32'), dmn_coords, node_size=150, display_mode="z", alpha=0.9, annotate=False, axes=ax[i])

fig.savefig(os.path.join(figure_path, "connectomes.pdf"))

plotting.plot_connectome(adj.astype('float32'), dmn_coords, node_size=150, display_mode="z", alpha=0.9, annotate=False, output_file=os.path.join(figure_path, "connectome.pdf"))




