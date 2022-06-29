# Main training script - trains distributedly accordi

import os
import hydra

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.nn.functional as F 
# import torch.optim as optim
# import torch.utils.data as data 

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# Custom imports 
from contrastive_learning.utils.logger import Logger
from contrastive_learning.datasets.dataset import get_dataloaders

class Workspace:
    # TODO: clean this code - it should be cfg: DictConfig (there should be less space)
    def __init__(self, cfg : DictConfig) -> None:
        print(f'Workspace config: {OmegaConf.to_yaml(cfg)}')

        # Initialize logger (wandb)
        wandb_exp_name = '-'.join(HydraConfig.get().run.dir.split('/')[-2:])
        self.logger = Logger(cfg, wandb_exp_name)

        # Create the checkpoint directory - it will be inside the hydra directory
        cfg.checkpoint_dir = os.path.join(HydraConfig.get().run.dir, 'models')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True) # Doesn't give an error if dir exists when exist_ok is set to True 
        
        # Set the world size according to the number of gpus
        cfg.num_gpus = torch.cuda.device_count()
        print(f"cfg.num_gpus: {cfg.num_gpus}")
        print()
        cfg.world_size = cfg.world_size * cfg.num_gpus

        # Calculate frame_interval
        cfg.frame_interval = cfg.fps * cfg.sec_interval

        # Set device and config
        # self.device = torch.device(torch.device(f'cuda:0') if torch.cuda.is_available() else "cpu")
        self.cfg = cfg

    def train(self, rank) -> None:
        print(f"INSIDE train: rank: {rank}")
        # Create default process group
        dist.init_process_group("gloo", rank=rank, world_size=self.cfg.world_size)
        dist.barrier() # Wait for all of the processes to start
        
        # Set the device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(self.cfg) # Sizes of train and  val loaders will be different
        print(f"Rank: {rank}, len(train_loader): {len(train_loader)}")

        cpn = hydra.utils.instantiate(self.cfg.agent).to(device)



        # Initialize the models and move the model to GPU
        # TODO: these might not also be completely good - a wrapper agent should be added here 
        # and both the encoder and trans should be used there not here 
        # encoder = hydra.utils.instantiate(self.cfg.encoder).to(device)
        # trans = hydra.utils.instantiate(self.cfg.trans).to(device)
        # print(f"encoder: {encoder}, trans: {trans}")
        # encoder = DDP(encoder, device_ids=[rank], output_device=rank)
        # trans = DDP(trans, device_ids=[rank], output_device=rank)

        # # Intialize the optimizer
        # parameters = list(encoder.parameters()) + list(trans.parameters())
        # optim = hydra.utils.instantiate(self.cfg.optimizer, params = parameters)

        # # Get the loss function
        # criterion = hydra.utils.instantiate(self.cfg.loss).to(device) # TODO: check this works

        # best_loss = torch.inf # We will keep the model with the lowest loss

        # Start the training






@hydra.main(version_base=None,config_path='configs', config_name = 'train')
def main(cfg : DictConfig) -> None:

    # TODO: check this and make it work when it's not distributed as well
    assert cfg.distributed is True, "Use script only to tran distributed"
    workspace = Workspace(cfg)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    
    print("Distributed training enabled. Spawning {} processes.".format(workspace.cfg.world_size))
    mp.spawn(workspace.train, nprocs=workspace.cfg.world_size)
    
if __name__ == '__main__':
    main()