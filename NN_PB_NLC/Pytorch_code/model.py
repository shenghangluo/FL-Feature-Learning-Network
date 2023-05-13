import torch
from torch import nn
import numpy as np
from scipy.io import loadmat

N = 894 # number of triplets per symbol
M = 2      # number of output symbol
def complex_multiply(x, y):
    xr = x[:, 0]
    xi = x[:, 1]
    yr = y[:, 0]
    yi = y[:, 1]
    return torch.stack([xr * yr - xi * yi, xr * yi + xi * yr], dim=1)

def complex_rotation(x,y):
    sinx = torch.sin(x[:,0])
    cosx = torch.cos(x[:,0])
    yr = y[:, 0]
    yi = y[:, 1]
    return torch.stack([cosx * yr - sinx * yi, cosx * yi + sinx * yr], dim=1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(N, 2),
            nn.LeakyReLU(0.5),
            nn.Linear(2, 10),
            nn.LeakyReLU(0.5),
            nn.Dropout(p=0.1),
            nn.Linear(10, 2),
        )

    def forward(self, x, Rx):
        self.AD_NL = self.DNN(x)
        output = Rx-self.AD_NL
        return output

    def get_AD(self):
        return self.AD_NL
