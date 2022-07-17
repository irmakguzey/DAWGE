import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from contrastive_learning.utils.losses import mse, infonce 

class SBFD: # State Based Forward Dynamics Agent (Pretty similar to CPN with small differences)
    def __init__(self,
                 pos_encoder, # Pos -> Embedding model is named as encoder as well
                 trans,
                 optimizer,
                 loss_fn) -> None:
        
        self.pos_encoder = pos_encoder 
        self.trans = trans 
        self.optimizer = optimizer 

        self.loss_type = loss_fn
        if loss_fn == "infonce":
            self.loss_fn = infonce
        elif loss_fn == "mse":
            self.loss_fn = mse 

    def to(self, device):
        self.pos_encoder.to(device)
        self.trans.to(device)
        self.device = device

    def train(self):
        self.pos_encoder.train()
        self.trans.train()
    
    def eval(self):
        self.pos_encoder.eval()
        self.trans.eval()

    def save(self, checkpoint_dir):
        torch.save(self.pos_encoder.state_dict(),
                   os.path.join(checkpoint_dir, 'pos_encoder.pt'),
                   _use_new_zipfile_serialization=False)
        torch.save(self.trans.state_dict(),
                   os.path.join(checkpoint_dir, 'trans.pt'),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader):
        # Set the training mode for both models
        self.train()

        # Save the train loss
        train_loss = 0.0

        for batch in train_loader:
            self.optimizer.zero_grad()
            pos, pos_next, action = [b.to(self.device) for b in batch]

            z, z_next = self.pos_encoder(pos), self.pos_encoder(pos_next) # b x z_dim 
            z_next_predict = self.trans(z, action)  # b x z_dim
            if self.loss_type == "mse":
                loss = self.loss_fn(z_next, z_next_predict)
            elif self.loss_type == "infonce":
                loss = self.loss_fn(z, z_next, z_next_predict) # TODO: infonce was changed so you should check this
            train_loss += loss.item()

            # Back prop
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters, 20)
            self.optimizer.step() 

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader):
        # Set the eval mode for both models
        self.eval()

        # Save the test loss
        test_loss = 0.0

        # Test for one epoch
        for batch in test_loader:
            obs, obs_next, action = [b.to(self.device) for b in batch]
            with torch.no_grad():
                z, z_next = self.pos_encoder(obs), self.pos_encoder(obs_next) # b x z_dim 
                z_next_predict = self.trans(z, action)  # b x z_dim
                if self.loss_type == "mse":
                    loss = self.loss_fn(z_next, z_next_predict)
                elif self.loss_type == "infonce":
                    loss = self.loss_fn(z, z_next, z_next_predict) # TODO: infonce was changed so you should check this
                test_loss += loss.item()

        return test_loss / len(test_loader)