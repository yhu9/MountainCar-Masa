import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

#DDQN NETWORK
class DDQN(nn.Module):
    def __init__(self,state_size=4,action_size=2):
        super(DDQN,self).__init__()

        #self.res18 = resnet.resnet18()
        self.h1 = nn.Linear(state_size,256)
        self.value = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,1)
                )

        self.action = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,action_size)
                )

    def forward(self,state):
        x = self.h1(state)
        v = self.value(x)
        a = self.action(x)
        q = v + (a - torch.mean(a))
        return q


