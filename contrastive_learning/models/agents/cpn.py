import hydra

import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Contrastive Predictive Network agent 
# Gets an encoder and a forward model and trains the models with the given loss

class CPN:
    def __init__(self,
                 encoder, 
                 trans, 
                 z_dim: int,
                 action_dim: int,
                 loss_fn,
                 optimizer) -> None:
        
        print(f"Encoder: {encoder}, trans: {trans}, z_dim: {z_dim}, action_dim: {action_dim}, loss_fn: {loss_fn}, optimizer: {optimizer}")

        