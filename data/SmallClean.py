from dataclasses import replace
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class RandomClean(Dataset):

    def __init__(self, train_data, train_labels):
        # Initialize data, download, etc.

        # find which random samples to use
        # rand_selection = np.random.choice(
        #     a=train_data.shape[0], size=1000, replace=False)
        rand_selection = torch.randperm(len(train_data))[:1000]
        print(rand_selection)
        self.train_data = train_data[rand_selection.to(dtype=torch.long)]
        self.train_labels = train_labels[rand_selection.to(dtype=torch.long)]
        self.n_samples = len(rand_selection)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples