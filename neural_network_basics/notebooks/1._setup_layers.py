# Create neural network class

import torch
import torch.nn as nn

class MyNet(nn.Module):

    # define constructor
    def __init__(self, input_size, hidden_size, output_size):
        # call nn.Module constructor
        super(MyNet, self).__init__()
        # define input layer for getting inputs
        self.input_layer = nn.Linear(input_size, hidden_size)
        # define output layer for producing output
        self.output_layer = nn.Linear(hidden_size, output_size)

    # define forward function
    def forward(self, x):
        # pass data x to input layer
        out = self.input_layer(x)
        # pass transformed data to next layer
        out = self.output_layer(out)

        return out
