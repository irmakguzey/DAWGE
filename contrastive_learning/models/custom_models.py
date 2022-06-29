import torch 
import torch.nn as nn

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # print("mean: {}".format(x.mean()))
        print(x.shape)
        return x

# Class for the forward linear model while calculating infonce loss
class Transition(nn.Module):
    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim 
        hidden_dim = 64
        self.model = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
    def forward(self, z, a):
        x = torch.cat((z,a), dim=-1)
        x = self.model(x)
        return x