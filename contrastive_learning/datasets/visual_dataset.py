import glob
import numpy as np
import os
import pickle
import random
import torch
import torch.utils.data as data 

from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader as loader 
from torchvision.utils import save_image

# Custom imports - preprocessing should be done when dataset is initialized
from contrastive_learning.datasets.preprocess import dump_video_to_images, create_pos_pairs
import contrastive_learning.utils.utils as utils

class VisualDataset:
    def __init__(self, data_dir: str,
                 frame_interval: int,
                 video_type: str = 'color') -> None:

        # Get the roots
        roots = glob.glob(f'{data_dir}/box_a_*') # TODO: change this in the future
        roots = sorted(roots)

        # Preprocess data for all the roots
        for root in roots:
            # dump_video_to_images(root) # TODO: check if you wanna add depth
            create_pos_pairs(root, frame_interval=frame_interval)

        # Traverse through the data and append path of all obs to one pos_pairs array
        self.pos_pairs = []
        for root in roots:
            pos_pair_path = os.path.join(root, f'{video_type}_pos_pairs.pkl')
            with open(pos_pair_path, 'rb') as f:
                self.pos_pairs += pickle.load(f) # We need all pos_pairs in the same order when we retrieve the data

        # manually guided std and means - TODO: change these!!
        self.action_mean, self.action_std = self.calculate_action_mean_std()

        self.transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.CenterCrop((480,480)), # TODO: Burda 480,480 yap bunu
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _get_image(self, path): 

        img = self.transform(loader(path))

        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index): 
        obs_file, obs_next_file, action = self.pos_pairs[index]

        obs = self._get_image(obs_file)
        obs_next = self._get_image(obs_next_file)

        # Normalize the actions
        action = (action - self.action_mean) / self.action_std

        return obs, obs_next, torch.FloatTensor(action)

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in animation class 

    def calculate_action_mean_std(self):
        # print("Calculating action mean and std")
        actions = np.zeros((len(self.pos_pairs), 2))
        for i in range(len(self.pos_pairs)):
            _, _, action = self.pos_pairs[i]
            actions[i,:] = action

        action_std = actions.std(axis=0)
        action_mean = actions.mean(axis=0)

        # print(f"Actions Mean: {action_mean}, Std: {action_std}")

        return action_mean, action_std


def get_dataloaders(cfg : DictConfig):
    # Load dataset - splitting will be done with random splitter
    dataset = VisualDataset(data_dir=cfg.data_dir, frame_interval=cfg.frame_interval, video_type=cfg.video_type)

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(dataset, 
                                             [train_dset_size, test_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))
    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=test_sampler is None,
                                    num_workers=cfg.num_workers, sampler=test_sampler)

    return train_loader, test_loader, dataset

def plot_data(data_dir:str, frame_interval:int, num_images:int = 16) -> None:

    # Get the data loaders
    dataset = VisualDataset(data_dir=data_dir, frame_interval=8)
    train_dset_size = int(len(dataset) * 0.8)
    val_dset_size = len(dataset) - train_dset_size
    # Random split the train and validation datasets
    train_dset, val_dset = data.random_split(dataset, 
                                             [train_dset_size, val_dset_size],
                                             generator=torch.Generator().manual_seed(43))
    train_loader = data.DataLoader(train_dset, batch_size=1, shuffle=True,
                                    num_workers=4, sampler=None)
    val_loader = data.DataLoader(val_dset, batch_size=1, shuffle=True,
                                    num_workers=4, sampler=None)
    # print(f"Train Dataset Size: {len(train_dset)}, Test Dataset Size: {len(val_dset)}, Train Loader Size: {len(train_loader)}, Test Loader Size: {len(val_loader)}")


    # Inverse transform to negate normalization in images
    inv_trans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                        std = [ 1., 1., 1. ]),
                                  ])

    imgs = np.zeros((num_images*2, 3, 480,480))
    i = 0
    for batch in train_loader:
        if i >= num_images:
            break
        obs, obs_next, action = [b for b in batch]
        # print('action: {}'.format(action))
        obs, obs_next = inv_trans(obs).numpy(), inv_trans(obs_next).numpy()
        # obs = utils.add_arrow(obs, action[0])
        
        imgs[2*i,:] = obs[:]
        imgs[2*i+1,:] = obs_next[:]
        i += 1

    imgs = torch.FloatTensor(imgs) # (n_image,3,480,480)
    save_image(imgs, os.path.join(data_dir, 'pos_pairs_exs.png'), nrow=8)


if __name__ == "__main__":
    cfg = OmegaConf.load('/home/irmak/Workspace/DAWGE/contrastive_learning/configs/train.yaml')
    plot_data(cfg.data_dir, cfg.frame_interval)


