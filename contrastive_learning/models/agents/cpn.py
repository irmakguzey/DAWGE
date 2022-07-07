import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from contrastive_learning.utils.losses import infonce

# Contrastive Predictive Network agent with infonce loss
# Gets an encoder and a forward model and trains the models with the given loss
class CPN:
    def __init__(self,
                 encoder, 
                 trans, 
                 optimizer,
                 loss_fn: str # will be a string to indicate the loss function to be used
                 ) -> None:
        
        # print(f'encoder: {encoder}, trans: {trans}, optimizer: {optimizer}, loss_fn: {loss_fn}')
        self.encoder = encoder 
        self.trans = trans 
        self.optimizer = optimizer 

        if loss_fn == "infonce":
            self.loss_fn = infonce

    def to(self, device):
        self.encoder.to(device)
        self.trans.to(device)
        self.device = device

    def train(self):
        self.encoder.train()
        self.trans.train()
    
    def eval(self):
        self.encoder.eval()
        self.trans.eval()

    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(),
                   os.path.join(checkpoint_dir, 'encoder.pt'),
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
            obs, obs_next, action = [b.to(self.device) for b in batch]

            # Get the loss - NOTE these parameters will need modificationafterwards
            loss = self.loss_fn(self.encoder, self.trans, obs, obs_next, action)
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
                loss = self.loss_fn(self.encoder, self.trans, obs, obs_next, action)
                test_loss += loss.item()

        return test_loss / len(test_loader)
    