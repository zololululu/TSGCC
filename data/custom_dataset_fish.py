"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import rand_bbox
import copy
import matplotlib.pyplot as plt


""" 
    AugmentedDataset
    Returns an ts together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            print('aug_dict==========================')
            self.ts_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.ts_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset.__getitem__(index)
        ts = x['ts']
        print("ts,shape: {}".format(ts.shape))
        sample = {}
        sample['ts'] = self.ts_transform.augment(ts)
        sample['ts_augmented'] = self.augmentation_transform.augment(ts)
        sample['index'] = index
        sample['target'] = x['target']
        return sample


""" 
    NeighborsDataset
    Returns an ts with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            print('aug_dict==========================')
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']

        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        # to_tensor = transforms.ToTensor()
        # output['origin'] = to_tensor(anchor['ts'])

        if self.anchor_transform is not None:
            # in selflabel step
            output['ts'] = self.anchor_transform.augment(anchor['ts'])
            output['ts_augmented'] = self.neighbor_transform.augment(anchor['ts'])

            # in scan step
            output['anchor'] = self.anchor_transform.augment(anchor['ts'])
            output['neighbor'] = self.neighbor_transform.augment(neighbor['ts'])
        else:
            # in selflabel step
            output['ts'] = anchor['ts']
            output['ts_augmented'] = anchor['ts']

            # in scan step
            output['anchor'] = anchor['ts']
            output['neighbor'] = neighbor['ts']

        output['anchor_neighbors_indices'] = torch.from_numpy(self.indices[index])
        output['neighbor_neighbors_indices'] = torch.from_numpy(self.indices[neighbor_index])
        output['target'] = anchor['target']
        output['index'] = index
        
        return output


