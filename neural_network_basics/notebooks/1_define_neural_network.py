# Create a feedforward neural network

import torch
import torch.nn as nn

class MyNet(nn.Module):

    # initialize the layers in constructor
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNet, self).__init__()

        # input layer of shape (input_size, hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # activation function
        self.relu = nn.ReLU()
        # output layer of shape (hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # define the forward pass
    def forward(self, x):
        # pass data x to first layer
        out = self.fc1(x)
        # apply activation function 
        out = self.relu(out)
        # pass the activated output to last layer
        out = self.fc2(out)

        # return output
        return out
    
# Define the neural network
model = MyNet(input_size=10, hidden_size=5, output_size=2)
input_data = torch.randn(1, 10)
output = model(input_data)