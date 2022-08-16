import glob
import numpy as np
import os
import pickle
# import random
import torch
import torch.utils.data as data 

from omegaconf import DictConfig, OmegaConf

# # Custom imports - preprocessing should be done when dataset is initialized
from contrastive_learning.datasets.preprocess import smoothen_corners, dump_pos_corners, dump_rvec_tvec

class StateDataset:
    def __init__(self, cfg: DictConfig, single_dir=False, single_dir_root=None) -> None:

        # Get the roots
        self.cfg = cfg
        roots = []
        if not single_dir:
            all_files = glob.glob(f'{cfg.data_dir}/*') # TODO: change this in the future
            for root in all_files:
                if os.path.isdir(root):
                    roots.append(root)
            roots = sorted(roots)
        else:
            roots = [single_dir_root]

        # Preprocess data for all the roots
        # for root in roots:
        #     smoothen_corners(root)
        #     dump_pos_corners(root, cfg.frame_interval)
        #     dump_rvec_tvec(root, cfg.frame_interval)

        print('DATASET POS_REF: {}'.format(cfg.pos_ref))

        # Little check about the reference of the model
        if not 'pos_ref' in self.cfg:
            self.cfg.pos_ref = 'global'

        # Traverse through the data and append all pos_corners
        self.pos_corners = []
        self.pos_rvec_tvec = []
        self.pos_mean_rots = []
        for root in roots:
            with open(os.path.join(root, f'pos_corners_fi_{self.cfg.frame_interval}.pickle'), 'rb') as f:
                self.pos_corners += pickle.load(f) # We need all pos_pairs in the same order when we retrieve the data

            with open(os.path.join(root, f'pos_rvec_tvec_fi_{self.cfg.frame_interval}.pickle'), 'rb') as f:
                self.pos_rvec_tvec += pickle.load(f)

            with open(os.path.join(root, f'pos_mean_rot_fi_{self.cfg.frame_interval}.pickle'), 'rb') as f:
                self.pos_mean_rots += pickle.load(f)

        # Calculate mins and maxs to normalize positions and actions
        if cfg.pos_type == 'corners':
            self.action_min, self.action_max, self.corner_min, self.corner_max = self.calculate_corners_mins_maxs()
            self.normalization_fn = self.normalize_corner 
            self.half_idx = 8
            self.position_arr = self.pos_corners
        elif cfg.pos_type == 'rvec_tvec':
            self.action_min, self.action_max, self.rvecs_min, self.rvecs_max, self.tvecs_min, self.tvecs_max = self.calculate_rvec_mins_maxs()
            self.normalization_fn = self.normalize_rvec_tvec
            self.half_idx = 6
            self.position_arr = self.pos_rvec_tvec
        elif cfg.pos_type == 'mean_rot': # Mean and rotation
            self.action_min, self.action_max, self.mean_min, self.mean_max = self.calculate_mean_rot_mins_maxs()
            self.rot_max, self.rot_min = np.pi, -np.pi # Rotation has always been given between [-pi,pi]
            self.normalization_fn = self.normalize_mean_rot 
            self.half_idx = 3 
            self.position_arr = self.pos_mean_rots


        print('self.action_min: {}, self.action_max: {}'.format(self.action_min, self.action_max))

    def __len__(self):
        return len(self.position_arr)

    def __getitem__(self, index): 
        curr_pos, next_pos, action = self.position_arr[index]

        # Normalize positions
        curr_pos = torch.FloatTensor(self.normalization_fn(curr_pos))
        next_pos = torch.FloatTensor(self.normalization_fn(next_pos))

        # Add reference
        ref_tensor = torch.zeros((int(curr_pos.shape[0]/2)))
        if self.cfg.pos_ref == 'dog':
            ref_tensor = curr_pos[self.half_idx:]
        elif self.cfg.pos_ref == 'box':
            ref_tensor = curr_pos[:self.half_idx]
        ref_tensor = ref_tensor.repeat(2)
        curr_pos -= ref_tensor 
        next_pos -= ref_tensor 

        # Normalize actions
        action = torch.FloatTensor(self.normalize_action(action))

        return curr_pos, next_pos, action

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in 

    def normalize_mean_rot(self, mean_rot): # Pos: [box_mean_x, box_mean_y, box_rot, dog_mean_x, dog_mean_y, dog_rot] (6,)
        mean_rot[:2] = (mean_rot[:2] - self.mean_min) / (self.mean_max - self.mean_min)
        mean_rot[2] = (mean_rot[2] - self.rot_min) / (self.rot_max - self.rot_min)

        mean_rot[3:5] = (mean_rot[3:5] - self.mean_min) / (self.mean_max - self.mean_min)
        mean_rot[5] = (mean_rot[5] - self.rot_min) / (self.rot_max - self.rot_min)

        return mean_rot

    def normalize_rvec_tvec(self, pos): # Pos: [box_rvec, box_tvec, dog_rvec, dog_tvec]
        pos[:3] = (pos[:3] - self.rvecs_min) / (self.rvecs_max - self.rvecs_min)
        pos[6:9] = (pos[6:9] - self.rvecs_min) / (self.rvecs_max - self.rvecs_min)

        pos[3:6] = (pos[3:6] - self.tvecs_min) / (self.tvecs_max - self.tvecs_min)
        pos[9:] = (pos[9:] - self.tvecs_min) / (self.tvecs_max - self.tvecs_min)

        return pos

    def normalize_corner(self, corner): # Corner.shape: 8.2
        corner = (corner - self.corner_min) / (self.corner_max - self.corner_min)
        return corner.flatten() # TODO: Check if this causes any problems - Returns (16)

    def normalize_action(self, action):
        return (action - self.action_min) / (self.action_max - self.action_min)

    def calculate_rvec_mins_maxs(self):
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

    def calculate_mean_rot_mins_maxs(self):
        means = np.zeros((len(self.pos_mean_rots), 2))
        actions = np.zeros((len(self.pos_mean_rots), 2))
        for i in range(len(self.pos_mean_rots)):
            means[i,:] = self.pos_mean_rots[i][0][:2]
            actions[i,0] = self.pos_mean_rots[i][2][0]
            actions[i,1] = self.pos_mean_rots[i][2][1]

        action_min, action_max = actions.min(axis=0), actions.max(axis=0)
        mean_min, mean_max = means.min(axis=0), means.max(axis=0)

        return action_min, action_max, mean_min, mean_max

    def calculate_corners_mins_maxs(self):
        corners = np.zeros((len(self.pos_corners), 8,2))
        actions = np.zeros((len(self.pos_corners), 2))
        for i in range(len(self.pos_corners)):
            corners[i,:] = self.pos_corners[i][0]
            actions[i,0] = self.pos_corners[i][2][0]
            actions[i,1] = self.pos_corners[i][2][1]

        action_min, action_max = actions.min(axis=0), actions.max(axis=0)

        corner_min, corner_max = corners.min(axis=(0,1)), corners.max(axis=(0,1))
        corner_min, corner_max = np.expand_dims(corner_min, axis=0), np.expand_dims(corner_max, axis=0)
        corner_min, corner_max = np.repeat(corner_min, 8, axis=0), np.repeat(corner_max, 8, axis=0)

        return action_min, action_max, corner_min, corner_max

    def denormalize_action(self, action): # action.shape: 2
        return (action * (self.action_max - self.action_min)) + self.action_min

    def denormalize_corner(self, corner): # corner.shape: (16)
        corner = corner.reshape((8,2))
        return (corner * (self.corner_max - self.corner_min)) + self.corner_min # Returns (8,2)

    def denormalize_pos_rvec_tvec(self, pos): # pos.shape: (12)
        pos[:3] = pos[:3] * (self.rvecs_max - self.rvecs_min) + self.rvecs_min 
        pos[6:9] = pos[6:9] * (self.rvecs_max - self.rvecs_min) + self.rvecs_min

        pos[3:6] = pos[3:6] * (self.tvecs_max - self.tvecs_min) + self.tvecs_min
        pos[9:] = pos[9:] * (self.tvecs_max - self.tvecs_min) + self.tvecs_min

        return pos # Returns (12)

    def denormalize_mean_rot(self, mean_rot): # Shape: (6,)
        mean_rot[:2] = mean_rot[:2] * (self.mean_max - self.mean_min) + self.mean_min
        mean_rot[2] = mean_rot[2] * (self.rot_max - self.rot_min) + self.rot_min

        mean_rot[3:5] = mean_rot[3:5] * (self.mean_max - self.mean_min) + self.mean_min 
        mean_rot[5] = mean_rot[5] * (self.rot_max - self.rot_min) + self.rot_min

        return mean_rot

if __name__ == '__main__':
    cfg = OmegaConf.load('/home/irmak/Workspace/DAWGE/contrastive_learning/configs/train.yaml')
    cfg.batch_size = 1
    dset = StateDataset(
        cfg = cfg
    )

    data_loader = data.DataLoader(dset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    batch = next(iter(data_loader))
    curr_pos, next_pos, action = [b for b in batch]
    print('curr_pos: {}\nnext_pos: {}\naction: {}'.format(
        curr_pos, next_pos, action
    ))
