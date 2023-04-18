# Create neural network class

import torch
import torch.nn as nn

class MyNet(nn.Module):

    # define constructor
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    # define forward function
    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(x)

        return out
