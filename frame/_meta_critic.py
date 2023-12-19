from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class TaskConfigNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TaskConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class Meta_Critic(nn.Module):
    def __init__(self, state_dim, neurons=(512, 256, 1),device=None):
        super(Meta_Critic, self).__init__()

        self.device = device
        self.l1 = nn.Linear(state_dim , neurons[0]).to(device)
        self.l2 = nn.Linear(neurons[0], neurons[1]).to(device)
        self.l3 = nn.Linear(neurons[1], neurons[2]).to(device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)   #可以换成x = nn.functional.softplus(self.fc3(x))
        return torch.mean(x)


class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)