import glob
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data 

from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader as loader 

# Custom imports - preprocessing should be done when dataset is initialized
from contrastive_learning.datasets.preprocess import dump_video_to_images, create_pos_pairs

class Dataset:
    def __init__(self, data_dir : str, frame_interval : int, video_type : str) -> None:

        # Get the roots
        roots = glob.glob(f'{data_dir}/*')
        roots = sorted(roots)

        # Preprocess data for all the roots
        for root in roots:
            dump_video_to_images(root) # TODO: check if you wanna add depth
            create_pos_pairs(root, frame_interval=frame_interval)

        # Traverse through the data and append path of all obs to one pos_pairs array
        self.pos_pairs = []
        for root in roots:
            pos_pair_path = os.path.join(root, f'{video_type}_pos_pairs.pkl')
            with open(pos_pair_path, 'rb') as f:
                self.pos_pairs += pickle.load(f) # We need all pos_pairs in the same order when we retrieve the data

        # manually guided std and means - TODO: change these!!
        self.action_mean = np.array([0, 0.35])
        self.action_std = np.array([0.09, 0.13])

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


def get_dataloaders(cfg : DictConfig) -> data.DataLoader:
    # Load dataset - splitting will be done with random splitter
    dataset = Dataset(data_dir=cfg.data_dir, frame_interval=cfg.frame_interval, video_type=cfg.video_type)

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    val_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, val_dset = data.random_split(dataset, 
                                             [train_dset_size, val_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))

    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    val_sampler = data.DistributedSampler(val_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    val_loader = data.DataLoader(val_dset, batch_size=cfg.batch_size, shuffle=val_sampler is None,
                                    num_workers=cfg.num_workers, sampler=val_sampler)

    return train_loader, val_loader 


