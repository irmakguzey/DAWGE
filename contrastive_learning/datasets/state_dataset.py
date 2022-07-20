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

class StateDataset: # TODO: Separate these two dataset according to the position
    def __init__(self, data_dir: str, single_dir=False) -> None:

        # Get the roots
        if not single_dir:
            roots = glob.glob(f'{data_dir}/box_marker_*') # TODO: change this in the future
            roots = sorted(roots)
        else:
            roots = [data_dir]

        # Preprocess data for all the roots
        # for root in roots:
        #     smoothen_corners(root)
        #     dump_pos_corners(root)

        # Traverse through the data and append all pos_corners
        # self.pos_corners = []
        self.pos_rvec_tvec = []
        for root in roots:
            # with open(os.path.join(root, 'pos_corners.pickle'), 'rb') as f:
            #     self.pos_corners += pickle.load(f) # We need all pos_pairs in the same order when we retrieve the data

            with open(os.path.join(root, 'pos_rvec_tvec.pickle'), 'rb') as f:
                self.pos_rvec_tvec += pickle.load(f)

        # Calculate mean and std to normalize corners and actions
        # self.action_mean, self.action_std, self.corner_mean, self.corner_std = self.calculate_means_stds()
        # self.action_min, self.action_max, self.corner_min, self.corner_max = self.calculate_mins_maxs()
    
        self.action_min, self.action_max, self.rvecs_min, self.rvecs_max, self.tvecs_min, self.tvecs_max = self.calculate_mins_maxs()

    def __len__(self):
        return len(self.pos_rvec_tvec)

    # def __getitem__(self, index): - getitem with box corners
    #     self.received_ids.append(index)
    #     curr_pos, next_pos, action = self.pos_corners[index]
    #     # TODO: make sure to change this if it doesn't work
        

    #     # Normalize the positions - TODO: If this works nicely then delete mean/std approach
    #     curr_pos = torch.FloatTensor((curr_pos - self.corner_min) / (self.corner_max - self.corner_min))
    #     next_pos = torch.FloatTensor((next_pos - self.corner_min) / (self.corner_max - self.corner_min))

    #     # Normalize the actions
    #     action = torch.FloatTensor((action - self.action_min) / (self.action_max - self.action_min))

    #     # return box_pos, dog_pos, next_box_pos, next_dog_pos, action
    #     return torch.flatten(curr_pos), torch.flatten(next_pos), action

    def __getitem__(self, index): 
        curr_pos, next_pos, action = self.pos_rvec_tvec[index]

        # Normalize the positions - TODO: If this works nicely then delete mean/std approach
        curr_pos = torch.FloatTensor(self._normalize_pos(curr_pos))
        next_pos = torch.FloatTensor(self._normalize_pos(next_pos))

        # Normalize the actions
        action = torch.FloatTensor((action - self.action_min) / (self.action_max - self.action_min))

        # return box_pos, dog_pos, next_box_pos, next_dog_pos, action
        return curr_pos, next_pos, action # TODO: Add index is just so that we could track

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in 

    def _normalize_pos(self, pos): # Pos: [box_rvec, box_tvec, dog_rvec, dog_tvec]
        pos[:3] = (pos[:3] - self.rvecs_min) / (self.rvecs_max - self.rvecs_min)
        pos[6:9] = (pos[6:9] - self.rvecs_min) / (self.rvecs_max - self.rvecs_min)

        pos[3:6] = (pos[3:6] - self.tvecs_min) / (self.tvecs_max - self.tvecs_min)
        pos[9:] = (pos[9:] - self.tvecs_min) / (self.tvecs_max - self.tvecs_min)

        return pos

    # def calculate_means_stds(self):
    #     corners = np.zeros((len(self.pos_corners), 8,2))
    #     actions = np.zeros((len(self.pos_corners), 2))
    #     for i in range(len(self.pos_corners)):
    #         corners[i,:] = self.pos_corners[i][0]
    #         actions[i,0] = self.pos_corners[i][2][0]
    #         actions[i,1] = self.pos_corners[i][2][1]

    #     action_mean, action_std = actions.mean(axis=0), actions.std(axis=0)
    #     corner_mean, corner_std = corners.mean(axis=(0,1)), corners.std(axis=(0,1))
    #     # Expand corner mean and std to be able to make element wise operations with them
    #     corner_mean, corner_std = np.expand_dims(corner_mean, axis=0), np.expand_dims(corner_std, axis=0)
    #     corner_mean, corner_std = np.repeat(corner_mean, 8, axis=0), np.repeat(corner_std, 8, axis=0) # 8: 4*2 (4 corners and 2 markers)

    #     return action_mean, action_std, corner_mean, corner_std

    def calculate_mins_maxs(self):
        rvecs = np.zeros((len(self.pos_rvec_tvec), 2, 3)) # Three axises for each corner - mean should be taken through 0th and 1st axes
        tvecs = np.zeros((len(self.pos_rvec_tvec), 2, 3))
        actions = np.zeros((len(self.pos_rvec_tvec), 2))

        for i in range(len(self.pos_rvec_tvec)):
            for j in range(2):
                rvecs[i,j,:] = self.pos_rvec_tvec[i][0][j*6:(j*6)+3] 
                tvecs[i,j,:] = self.pos_rvec_tvec[i][0][j*6+3:(j+1)*6]

            actions[i,0] = self.pos_rvec_tvec[i][2][0]
            actions[i,1] = self.pos_rvec_tvec[i][2][1]

        rvecs_min, rvecs_max = rvecs.min(axis=(0,1)), rvecs.max(axis=(0,1))
        tvecs_min, tvecs_max = tvecs.min(axis=(0,1)), tvecs.max(axis=(0,1))
        action_min, action_max = actions.min(axis=0), actions.max(axis=0)

        return action_min, action_max, rvecs_min, rvecs_max, tvecs_min, tvecs_max

    # def calculate_mins_maxs(self):
        # corners = np.zeros((len(self.pos_corners), 8,2))
        # actions = np.zeros((len(self.pos_corners), 2))
        # for i in range(len(self.pos_corners)):
        #     corners[i,:] = self.pos_corners[i][0]
        #     actions[i,0] = self.pos_corners[i][2][0]
        #     actions[i,1] = self.pos_corners[i][2][1]

        # action_min, action_max = actions.min(axis=0), actions.max(axis=0)

        # corner_min, corner_max = corners.min(axis=(0,1)), corners.max(axis=(0,1))
        # corner_min, corner_max = np.expand_dims(corner_min, axis=0), np.expand_dims(corner_max, axis=0)
        # corner_min, corner_max = np.repeat(corner_min, 8, axis=0), np.repeat(corner_max, 8, axis=0)

        # return action_min, action_max, corner_min, corner_max

    def denormalize_action(self, action): # action.shape: 2
        # return (action * self.action_std) + self.action_mean
        return (action * (self.action_max - self.action_min)) + self.action_min

    def denormalize_corner(self, corner): # corner.shape: (16)
        corner = corner.reshape((8,2))
        # return (corner * self.corner_std) + self.corner_mean
        return (corner * (self.corner_max - self.corner_min)) + self.corner_min

    def denormalize_pos_rvec_tvec(self, pos): # pos.shape: (12)
        pos[:3] = pos[:3] * (self.rvecs_max - self.rvecs_min) + self.rvecs_min 
        pos[6:9] = pos[6:9] * (self.rvecs_max - self.rvecs_min) + self.rvecs_min

        pos[3:6] = pos[3:6] * (self.tvecs_max - self.tvecs_min) + self.tvecs_min
        pos[9:] = pos[9:] * (self.tvecs_max - self.tvecs_min) + self.tvecs_min

        return pos


# if __name__ == '__main__':
#     cfg = OmegaConf.load('/home/irmak/Workspace/DAWGE/contrastive_learning/configs/train.yaml')
#     cfg.batch_size = 1
#     dset = StateDataset(
#         data_dir = cfg.data_dir
#     )

#     data_loader = data.DataLoader(dset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
#     batch = next(iter(data_loader))
#     curr_pos, next_pos, action, _ = [b for b in batch]
#     print('curr_pos: {}, next_pos: {}, action: {}'.format(
#         curr_pos, next_pos, action
#     ))
    # print(dset.getitem(0))
    # print(len(dset))

    # train_loader, test_loader, _, _ = get_dataloaders(cfg)
    # print(len(train_loader))
