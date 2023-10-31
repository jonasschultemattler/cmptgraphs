import numpy as np
from dataloader import DisseminationDataloader
from brain_dataloader2 import BrainDevelopementDataloader

path = "../../datasets/brains"
dataset = "development"

roi = "atlas"
n_subjects = 100

# np.random.seed(1)

dataloader = BrainDevelopementDataloader(dataset, path, n_subjects, roi)
dataloader.savetxt()





# path = "../../datasets"
# dataset = "infectious_ct1"


# dataloader = DisseminationDataloader(dataset, path)
# tgraphs, labels = dataloader.load_temporal_graphs()
# dataloader.save(tgraphs, labels)
# tgraphs, labels = dataloader.load()
# dataloader.save_temporal_graphs(tgraphs, labels)
