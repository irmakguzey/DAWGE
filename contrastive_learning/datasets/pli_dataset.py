import glob
import numpy as np
import os
import pickle
# import random
import torch
import torch.utils.data as data 

from omegaconf import DictConfig, OmegaConf

# # Custom imports - preprocessing should be done when dataset is initialized
from omegaconf import DictConfig, OmegaConf
from contrastive_learning.datasets.preprocess import smoothen_corners, dump_pos_corners
# import contrastive_learning.utils.utils as utils

class Dataset:
    def __init__(self, data_dir: str) -> None:

        # Get the roots
        roots = glob.glob(f'{data_dir}/box_marker_*') # TODO: change this in the future
        roots = sorted(roots)

        # Preprocess data for all the roots
        # for root in roots:
        #     smoothen_corners(root)
        #     dump_pos_corners(root)

        # Traverse through the data and append all pos_corners
        self.pos_corners = []
        for root in roots:
            with open(os.path.join(root, 'pos_corners.pickle'), 'rb') as f:
                self.pos_corners += pickle.load(f) # We need all pos_pairs in the same order when we retrieve the data

        # Calculate mean and std to normalize corners and actions
        self.action_mean, self.action_std, self.corner_mean, self.corner_std = self.calculate_means_stds()

    
    def __len__(self):
        return len(self.pos_corners)

    def __getitem__(self, index): 
        # box_pos, dog_pos, next_box_pos, next_dog_pos, action = self.pos_corners[index]
        curr_pos, next_pos, action = self.pos_corners[index]

        # Normalize the positions
        curr_pos = torch.FloatTensor((curr_pos - self.corner_mean) / self.corner_std)
        next_pos = torch.FloatTensor((next_pos - self.corner_mean) / self.corner_std)

        # Normalize the actions
        action = torch.FloatTensor((action - self.action_mean) / self.action_std)

        # return box_pos, dog_pos, next_box_pos, next_dog_pos, action
        return torch.flatten(curr_pos), torch.flatten(next_pos), action

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in 

    def calculate_means_stds(self):
        corners = np.zeros((len(self.pos_corners), 8,2))
        actions = np.zeros((len(self.pos_corners), 2))
        for i in range(len(self.pos_corners)):
            corners[i,:] = self.pos_corners[i][0]
            actions[i,0] = self.pos_corners[i][2][0]
            actions[i,1] = self.pos_corners[i][2][1]

        action_mean, action_std = actions.mean(axis=0), actions.std(axis=0)
        corner_mean, corner_std = corners.mean(axis=(0,1)), corners.std(axis=(0,1))
        # Expand corner mean and std to be able to make element wise operations with them
        corner_mean, corner_std = np.expand_dims(corner_mean, axis=0), np.expand_dims(corner_std, axis=0)
        corner_mean, corner_std = np.repeat(corner_mean, 8, axis=0), np.repeat(corner_std, 8, axis=0) # 8: 4*2 (4 corners and 2 markers)

        return action_mean, action_std, corner_mean, corner_std

    def denormalize_action(self, action):
        return (action * self.action_std) + self.action_mean

def get_dataloaders(cfg : DictConfig):
    # Load dataset - splitting will be done with random splitter
    dataset = Dataset(data_dir=cfg.data_dir)

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(dataset, 
                                             [train_dset_size, test_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))
    # print('len(train_dset): {}'.format(len(train_dset)))
    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=test_sampler is None,
                                    num_workers=cfg.num_workers, sampler=test_sampler)

    return train_loader, test_loader, dataset

if __name__ == '__main__':
    cfg = OmegaConf.load('/home/irmak/Workspace/DAWGE/contrastive_learning/configs/train.yaml')
    dset = Dataset(
        data_dir = cfg.data_dir
    )

    print(dset.getitem(0))
    print(len(dset))

    train_loader, test_loader, _, _ = get_dataloaders(cfg)
    print(len(train_loader))
