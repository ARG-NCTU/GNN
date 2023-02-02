import torch_geometric
import torch.nn as nn
import torch

from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_cluster import radius_graph


class GNN_Actor(nn.Module):
    def __init__(self, encode_feature, state_feature, action_feature):
        super(GNN_Actor, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(encode_feature + state_feature, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Tanh()
                )
    # def forward(self, encode_input, state_input):
        # x = torch.cat([latent_out, state_input], dim=1)
        # out = self.main(x)
    def forward(self, encode_input):
        out = self.main(encode_input)
        return out

class GNN_Critic(nn.Module):
    def __init__(self, encode_feature, state_feature, action_feature):
        super(GNN_Critic, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(encode_feature + state_feature + action_feature, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
                )
    # def forward(self, encode_input, state_input, action):
        # x1 = torch.cat([encode_input, state_input], dim=1)
        # x2 = torch.cat([x1, action], dim=1)
        # out = self.main(x2)
    def forward(self, encode_input, action):
        out = self.main(torch.cat((encode_input, action), 1))
        return out


        

    # def reset_parameters(self):
    #     torch_geometric.nn.inits.reset(self.nn)

    # def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
    #     return self.propagate(edge_index, x=x, size=None)

    # def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
    #     return self.nn(x_j - x_i)

    # def __repr__(self):
    #     return "{}(nn={})".format(self.__class__.__name__, self.nn)