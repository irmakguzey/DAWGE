# Main training script - trains distributedly accordi

import os
import hydra
import logging
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm 


# Custom imports 
from contrastive_learning.utils.logger import Logger
from contrastive_learning.datasets.dataset import get_dataloaders


class Workspace:
    # TODO: clean this code - it should be cfg: DictConfig (there should be less space)
    def __init__(self, cfg : DictConfig) -> None:
        print(f'Workspace config: {OmegaConf.to_yaml(cfg)}')

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Create the checkpoint directory - it will be inside the hydra directory
        cfg.checkpoint_dir = os.path.join(self.hydra_dir, 'models')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True) # Doesn't give an error if dir exists when exist_ok is set to True 
        
        # Set the world size according to the number of gpus
        cfg.num_gpus = torch.cuda.device_count()
        print(f"cfg.num_gpus: {cfg.num_gpus}")
        print()
        cfg.world_size = cfg.world_size * cfg.num_gpus

        # Set device and config
        self.cfg = cfg

    def train(self, rank) -> None:
        # Create default process group
        dist.init_process_group("gloo", rank=rank, world_size=self.cfg.world_size)
        dist.barrier() # Wait for all of the processes to start
        
        # Set the device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print(f"INSIDE train: rank: {rank} - device: {device}")

        # Get dataloaders
        # TODO: maybe create the datasets in the configs
        train_loader, test_loader, _, _ = get_dataloaders(self.cfg) # Sizes of train and  val loaders will be different

        # Initialize the encoder and the trans
        encoder = hydra.utils.instantiate(self.cfg.encoder).to(device)
        trans = hydra.utils.instantiate(self.cfg.trans,
                                        z_dim=self.cfg.z_dim,
                                        action_dim=self.cfg.action_dim).to(device)
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False) # To fix the inplace error https://github.com/pytorch/pytorch/issues/22095 
        trans = DDP(trans, device_ids=[rank], output_device=rank, broadcast_buffers=False)

        # Initialize the optimizer
        parameters = list(encoder.parameters()) + list(trans.parameters())
        optimizer = hydra.utils.instantiate(self.cfg.optimizer,
                                            params = parameters,
                                            lr = self.cfg.lr,
                                            weight_decay = self.cfg.weight_decay)

        # Initialize the total agent
        agent = hydra.utils.instantiate(self.cfg.agent,
                                        encoder=encoder,
                                        trans=trans,
                                        optimizer=optimizer)
        agent.to(device)

        best_loss = torch.inf 

        # Logging
        if rank == 0:
            pbar = tqdm(total=self.cfg.train_epochs)
            # Initialize logger (wandb)
            wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
            self.logger = Logger(self.cfg, wandb_exp_name, out_dir=self.hydra_dir)


        # Start the training
        for epoch in range(self.cfg.train_epochs):
            # Distributed settings
            if self.cfg.distributed:
                train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            # Train the models for one epoch
            train_loss = agent.train_epoch(train_loader)

            if self.cfg.distributed:
                dist.barrier()

            if rank == 0: # Will only print after everything is finished
                pbar.set_description(f'Epoch {epoch}, Train loss: {train_loss:.5f}, Best loss: {best_loss:.5f}')
                pbar.update(1) # Update for each batch

            # Logging
            if rank == 0 and epoch % self.cfg.log_frequency == 0:
                # print(f'logging in rank: {rank}')
                self.logger.log({'epoch': epoch,
                                 'train loss': train_loss})

            # Testing and saving the model
            if epoch % self.cfg.save_frequency == 0:
                # Test for one epoch
                test_loss = agent.test_epoch(test_loader)
                
                # Get the best loss
                if test_loss < best_loss:
                    best_loss = test_loss
                    agent.save(self.cfg.checkpoint_dir)

                # Logging
                if rank == 0:
                    pbar.set_description(f'Epoch {epoch}, Test loss: {test_loss:.5f}')
                    self.logger.log({'epoch': epoch,
                                    'test loss': test_loss})
                    self.logger.log({'epoch': epoch,
                                    'best loss': best_loss})

        if rank == 0: 
            pbar.close()

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