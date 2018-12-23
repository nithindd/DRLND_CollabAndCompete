import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_in_dim, hidden_out_dim):
        super(Actor, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim,hidden_in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.bn3 = nn.BatchNorm1d(hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,action_dim)
        self.nonlin = f.leaky_relu
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        x = self.bn1(state)
        x =  self.nonlin(self.fc1(x))
        # x = self.bn2(x)
        x = self.nonlin(self.fc2(x))
        # x = self.bn3(x)
        x = self.fc3(x)
        return f.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_in_dim, hidden_out_dim):
        super(Critic, self).__init__()
       
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim+action_dim,hidden_out_dim)
        self.bn2 = nn.BatchNorm1d(hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,1)
        self.nonlin = f.leaky_relu

    def forward(self, state, action):
        x = self.bn1(state)
        h1 = self.nonlin(self.fc1(x))
        x = torch.cat((h1, action), dim=1)
        h2 = self.nonlin(self.fc2(x))
        h3 = (self.fc3(h2))
      
        return h3
