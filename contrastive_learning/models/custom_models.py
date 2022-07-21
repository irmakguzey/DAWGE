import torch 
import torch.nn as nn

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # print("mean: {}".format(x.mean()))
        print(x.shape)
        return x

# Simple linear layer to map the positions to embeddings
class PosToEmbedding(nn.Module): # Model to simply map raw positions to embeddings - it will be used in infonce loss with state based dataset
    def __init__(self, input_dim, hidden_dim, out_dim): # Gets the posisionts 
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pos):
        z = self.model(pos)
        return z

# Class for the forward linear model while calculating infonce loss
class Transition(nn.Module):
    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.a_repeatition = int(action_dim / 2)
        hidden_dim = 64
        # self.model = nn.Sequential(
        #     nn.Linear(z_dim + action_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2*hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(2*hidden_dim, 4*hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(4*hidden_dim, 2*hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(2*hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, z_dim)
        # )
        self.model = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, z_dim)
        )
        
    def forward(self, z, a):
        curr_a = a
        for _ in range(self.a_repeatition-1):
            a = torch.cat((a,curr_a), dim=-1)
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
            nn.Linear(input_dim*2, hidden_dim), # input_dim*2: For current and next (so in total 32)
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), action_dim)
        )        

    def forward(self, curr_pos, next_pos):
        x = torch.cat((curr_pos, next_pos), dim=-1)
        x = self.model(x)
        return x