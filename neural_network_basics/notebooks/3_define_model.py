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
        # define activation function    
        self.relu = nn.ReLU()
        # define output layer for producing output
        self.output_layer = nn.Linear(hidden_size, output_size)

    # define forward function
    def forward(self, x):
        # pass data x to input layer
        out = self.input_layer(x)
        # apply activation function to data
        out = self.relu(x)
        # pass activated output to next layer
        out = self.output_layer(x)

        return out

# define neural network model
model = MyNet(input_size=10, hidden_size=5, output_size=2)
input_data = torch.randn(1, 10)
output = model(input_data)