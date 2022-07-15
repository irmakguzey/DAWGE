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
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, z_dim)
        )
        
    def forward(self, z, a):
        x = torch.cat((z,a), dim=-1)
        x = self.model(x)
        return x

# Class for the forward inverse model where it gets the dog's position, 
# box's position, their next position and returns the action applied between
class LinearInverse(nn.Module):
    # NOTE: For now the input will be [robot_rotation, box_rotation, distance_bw]
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )        

    def forward(self, curr_pos, next_pos):
        x = torch.cat((curr_pos, next_pos), dim=-1)
        x = self.model(x)
        return x