import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    """ A standard in_tim-64-64-out_dim Feed Forward Neural Network"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=32)
        self.layer4 = nn.Linear(in_features=32, out_features=24)
        self.layer5 = nn.Linear(in_features=24, out_features=24)
        self.layer6 = nn.Linear(in_features=24, out_features=out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        activation4 = F.relu(self.layer4(activation3))
        activation5 = F.relu(self.layer5(activation4))
        output = self.layer6(activation5)
        return output
