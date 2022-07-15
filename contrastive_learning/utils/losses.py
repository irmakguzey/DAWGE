# Script to implement loss functions 

import numpy as np
import torch
import torch.nn.functional as F 


def mse(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)

def infonce(z, z_next, z_next_predict):
    bs = z.shape[0] 

    neg_dot_products = torch.mm(z_next_predict, z.t()) # b x b
    neg_dists = -((z_next_predict ** 2).sum(1).unsqueeze(1) - 2*neg_dot_products + (z ** 2).sum(1).unsqueeze(0))

    idxs = np.arange(bs)
    neg_dists[idxs, idxs] = float('-inf') # b x b+1

    pos_dot_products = (z_next * z_next_predict).sum(1) # b
    pos_dists = -((z_next**2).sum(1) - 2*pos_dot_products + (z_next_predict ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1) # b x 1 

    dists = torch.cat((neg_dists, pos_dists), dim=1)
    dists = F.log_softmax(dists, dim=1)
    loss = -dists[:,-1].mean()

    return loss