import os
import hydra
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 
from typing import Tuple, Optional, List

from contrastive_learning.tests.plotting import plot_corners, plot_rvec_tvec, plot_mean_rot

# Custom imports 
from contrastive_learning.utils.losses import mse, l1

class Diffusion:
    def __init__(self,
                 eps_model: nn.Module,
                 optimizer,
                 n_steps,
                 dataset,
                 checkpoint_dir) -> None: # We are getting the current cfg to plot the samples correctly
        
        self.eps_model = eps_model 
        self.optimizer = optimizer 
        self.dataset = dataset # Dataset will only be used for plotting samples
        self.checkpoint_dir = checkpoint_dir
        self.test_epoch_num = 0 # This is for keeping track of the images

        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps # Number of steps to noise and denoise the data
        self.sigma2 = self.beta

    def to(self, device):
        self.device = device

        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sigma2 = self.sigma2.to(device)

        self.eps_model = self.eps_model.to(device)

    def train(self):
        self.eps_model.train()

    def eval(self):
        self.eps_model.eval()

    def save(self, checkpoint_dir): # NOTE: You might add this to regular Agent function
        torch.save(self.eps_model.state_dict(),
                   os.path.join(checkpoint_dir, 'eps_model.pt'),
                   _use_new_zipfile_serialization=False)

    # Method to be used in testing 
    # Testing loss should be calculated with this method rather than diffusion loss
    # Because in testing we don't care about diffusion loss at all
    # TODO: Add the images to wandb
    def sample(self, batch, save_plot=False): # If save_plot is given True checkpoint_dir should not be None
        x0, xnext0, a = [b.to(self.device) for b in batch]
        xt = torch.randn((x0.shape), device=x0.device)
        for t_ in range(self.n_steps):
            curr_t = self.n_steps - t_ - 1
            t = xt.new_full((xt.shape[0],), curr_t, dtype=torch.long)
            xt = self.p_sample(xt, t, x0, a)

        sample_loss = F.mse_loss(xnext0, xt)

        if save_plot:
            bs = x0.shape[0]
            pos_dim = int(x0.shape[1] / 2)
            # print('pos_dim: {}'.format(pos_dim))
            if pos_dim == 8: # Pos type is corners
                plotting_fn = plot_corners
                denormalize_fn = self.dataset.denormalize_corner
            elif pos_dim == 6: # Pos type is rotational and translational vectors
                plotting_fn = plot_rvec_tvec
                denormalize_fn = self.dataset.denormalize_pos_rvec_tvec # NOTE: This will def cause some problems
            elif pos_dim == 3: # Pos type is just the mean and rotation of the box
                plotting_fn = plot_mean_rot
                denormalize_fn = self.dataset.denormalize_mean_rot

            ncols = 10
            nrows = math.ceil(bs / ncols)
            fig, axs = plt.subplots(figsize=(10*ncols, 10*nrows), nrows=nrows, ncols=ncols)
            fig.suptitle('Samples in {}th Test'.format(self.test_epoch_num))

            # Denormalize all the positions
            for i in range(bs):
                # x0_curr = denormalize_fn(x0[i].cpu().detach().numpy())
                xnext0_curr = denormalize_fn(xnext0[i].cpu().detach().numpy())
                xt_curr = denormalize_fn(xt[i].cpu().detach().numpy())

                # Plot the denormalized corners
                axs_row = int(i / ncols)
                axs_col = int(i % ncols)
                axs[axs_row, axs_col].set_title("Data {} in the batch".format(i))
                _, frame_axis = plotting_fn(axs[axs_row, axs_col], xnext0_curr, color_scheme=1)
                plotting_fn(axs[axs_row, axs_col], xt_curr, use_frame_axis=True, frame_axis=frame_axis, color_scheme=2)

        # Save the saving plot
        plt.savefig(os.path.join(self.checkpoint_dir, 'samples_{}.png'.format(self.test_epoch_num)))

        return sample_loss

    def train_epoch(self, train_loader): # NOTE: train_loader will give applied actions as well
        # Set the training mode for both models
        self.train()

        # Save the train loss
        train_loss = 0.0

        for batch in train_loader:
            self.optimizer.zero_grad()
            # curr_pos, next_pos, action = [b.to(self.device) for b in batch]
            x0, xnext0, a = [b.to(self.device) for b in batch]
            
            # Calculate the diffusion loss
            loss = self.loss(x0, xnext0, a)

            train_loss += loss.item()

            # Backprop
            loss.backward()
            self.optimizer.step() 

        return train_loss / len(train_loader)


    def test_epoch(self, test_loader):
        # Set the eval mode for both models
        self.eval()
        self.test_epoch_num += 1 # This is only to keep a track of samples

        # Save the test loss
        test_loss = 0.0

        # Test for one epoch
        for batch in test_loader:
            with torch.no_grad():
                sample_loss = self.sample(batch)
                test_loss += sample_loss.item()

        # Sample with the last batch
        self.sample(batch, save_plot=True)

        return test_loss / len(test_loader)

    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1)

    # q(xt|x0,t) - gives noised xts for different ts - noises them starting from completely clean data
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x0 # This multiplication here makes everything dirty 
        var = 1 - self.gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, a: torch.Tensor):
        eps_theta = self.eps_model(xt, t, x0, a) # This will for sure be complete noise
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gather(self.sigma2, t) # (1 - self.alpha)

        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 5.) * eps

    # x0 and xnext0 are supposed to be clean tensors
    def loss(self, x0: torch.Tensor, xnext0: torch.Tensor, a: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = xnext0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=xnext0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(xnext0)

        # Noise the clean next state
        xnextt = self.q_sample(xnext0, t, eps=noise)
        # Learn the noise applied to xnext0 
        eps_theta = self.eps_model(xnextt, t, x0, a) # Prediction of the next state should have the current state
        return F.mse_loss(noise, eps_theta)