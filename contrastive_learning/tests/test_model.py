
import numpy as np
import os
import torch

from collections import OrderedDict
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP

from contrastive_learning.models.pretrained_models import resnet18
from contrastive_learning.models.custom_models import LinearInverse

# Script to have methods for testing models
# Method to take encoder and test_loader passes all of them through the encoder and save the 
# representations in the given directory (create the directory if not exists)
def save_all_embeddings(device, len_dset, z_dim, encoder, train_loader, out_dir):
    pbar = tqdm(total=len(train_loader))

    # Get all the embeddings
    all_z = np.zeros((len_dset, z_dim))
    all_z_next = np.zeros((len_dset, z_dim))
    ep = 0
    for batch in train_loader: 
        obs, obs_next, _ = [b.to(device) for b in batch]
        bs = obs.shape[0]

        z, z_next = encoder(obs), encoder(obs_next)
        all_z[ep*bs:(ep+1)*bs, :] = z.cpu().detach().numpy()
        all_z_next[ep*bs:(ep+1)*bs, :] = z_next.cpu().detach().numpy()

        ep += 1
        pbar.update(1)
    pbar.close()

    # Dump them in out_dir as a numpy array
    with open(os.path.join(out_dir, 'all_z.npy'), 'wb') as f:
        np.save(f, all_z)
    with open(os.path.join(out_dir, 'all_z_next.npy'), 'wb') as f:
        np.save(f, all_z_next)

# Method that takes an encoder and an observation, gets a representation 
# Then finds the K closest embeddings 
def get_closest_embeddings(out_dir, encoder, obs, k):
    with open(os.path.join(out_dir, 'all_z.npy'), 'rb') as f:
        all_z = np.load(f)

    z = encoder(obs).cpu().detach().numpy()
    dist = np.linalg.norm(all_z - z, axis=1)
    closest_z_idx = np.argsort(dist)[:k]
    print('closest_z_idx.shape: {}'.format(closest_z_idx.shape))

    return closest_z_idx

def load_encoder(device, encoder_path, encoder_type:str):
    if encoder_type == 'resnet18':
        encoder = resnet18(pretrained=False) # We will set the weights ourselves
    
    state_dict = torch.load(encoder_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    encoder.load_state_dict(new_state_dict)

    # Turn it into DDP - it was saved that way
    encoder = DDP(encoder.to(device), device_ids=[0])

    return encoder

def load_lin_model(cfg, device, model_path):
    lin_model = LinearInverse(cfg.pos_dim*2, cfg.action_dim, cfg.hidden_dim)
    
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    lin_model.load_state_dict(new_state_dict)

    # Turn it into DDP - it was saved that way
    lin_model = DDP(lin_model.to(device), device_ids=[0])

    return lin_model