import os
import numpy as np
import torch
import numpy
from torch.utils.data import Dataset
from utils.mypath import MyPath
import pandas as pd


class FISH(Dataset):

    def __init__(self, root='./path/to', train=True, transform=None, datasetname=None):

        super(FISH, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        if self.train:
            # file_name = os.path.join(datasetname + '_noise', datasetname + '_TRAIN')
            file_name = os.path.join(datasetname, datasetname + '_TRAIN')
        else:
            # file_name = os.path.join(datasetname + '_noise', datasetname + '_TEST')
            file_name = os.path.join(datasetname, datasetname + '_TEST')

        # print(file_name)
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        file_path = os.path.join(self.root, file_name)
        with open(file_path, 'rb') as f:
            data = pd.read_csv(file_path, usecols=None, header=None)
            data = data.values
            self.targets = data[:, 0:1].reshape(1, -1)[0]
            self.data = data[:, 1:]
            self.classes = numpy.unique(self.targets).tolist()

    def __getitem__(self, index):
        ts, target = self.data[index], self.targets[index]
        ts_size = (ts.shape[0])

        if self.transform is not None:
            ts = self.transform.augment(ts)

        out = {'ts': ts, 'target': target, 'meta': {'im_size': ts_size, 'index': index}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
