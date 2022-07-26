import os
import hydra
import torch
# import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from contrastive_learning.utils.losses import mse, l1

# Agent to get current and next position and predict the action applied
# It will learn supervised way
# Predictive Linear Inverse Model 
class PLI:
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn: str,
                 use_encoder: bool, # This is not used for now - if we were to add encoder could be useful
                 ) -> None:

        self.model = model 
        self.optimizer = optimizer

        if loss_fn == 'mse': # TODO: try different loss functions?
            self.loss_fn = mse 
        elif loss_fn == 'l1':
            self.loss_fn = l1

    def to(self, device):
        self.device = device 
        self.model.to(device)
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, checkpoint_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(checkpoint_dir, 'lin_model.pt'),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader): # NOTE: train_loader will give applied actions as well
        # Set the training mode for both models
        self.train()

        # Save the train loss
        train_loss = 0.0

        for batch in train_loader:
            self.optimizer.zero_grad()
            curr_pos, next_pos, action = [b.to(self.device) for b in batch]

            # Find the predicted action
            pred_action = self.model(curr_pos, next_pos)

            # Get the supervised
            loss = self.loss_fn(action, pred_action)
            train_loss += loss.item()

            # Backprop
            loss.backward()
            self.optimizer.step() 

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader):
        # Set the eval mode for both models
        self.eval()

        # Save the test loss
        test_loss = 0.0

        # Test for one epoch
        for batch in test_loader:
            curr_pos, next_pos, action = [b.to(self.device) for b in batch]
            pred_action = self.model(curr_pos, next_pos)
            with torch.no_grad():
                loss = self.loss_fn(action, pred_action)
                test_loss += loss.item()

        return test_loss / len(test_loader)
    