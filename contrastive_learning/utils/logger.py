from tokenize import String
import wandb

from omegaconf import DictConfig, OmegaConf

# TODO: maybe add logger from logging library?
class Logger:
    def __init__(self, cfg : DictConfig, exp_name : String) -> None:

        # Initialize the wandb experiment
        wandb.init(project="dawge", 
            name=exp_name,
            config = OmegaConf.to_container(cfg, resolve=True), 
            settings=wandb.Settings(start_method="thread"))

    def log(self, msg):
        wandb.log(msg)