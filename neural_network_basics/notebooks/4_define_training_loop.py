import torch 
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Generate 100 random data points
np.random.seed(42)
X = np.random.rand(100, 2)

# Define the labels based on a simple classification rule
y = (X[:, 0] + X[:, 1] > 1).astype(int)

class MyNet(nn.Module):

    # define the constructor
    def __init__(self, input_size, hidden_size, output_size):
        # call nn.Module constructor
        super(MyNet, self).__init__()
        # define the input layer for getting inputs
        self.input_layer = nn.Linear(input_size, hidden_size)
        # define the hidden layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        # define the activation function
        self.relu = nn.ReLU()
        # define output layer for producing output
        self.output_layer = nn.Linear(hidden_size, output_size)
        # activation function for producing output
        self.sigmoid = nn.Sigmoid()

    # define forward faction
    def forward(self, x):
        # pass the data x to input layer
        out = self.input_layer(x)
        # pass the transformed data to next layer
        out = self.hidden_layer(out)
        # apply activation function to data
        out = self.relu(out)
        # pass the activated output to next layer
        out = self.output_layer(out)
        # pass predictions to sigmoid
        out = self.sigmoid(out)

        return out
    
model = MyNet(input_size=10, hidden_size=5, output_size=2)
input_data = torch.randn(1, 10)
output = model(input_data)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# define loss function/criterion
criterion = nn.BCELoss()

# define accuracy function
def binary_accuracy(preds, y):
    """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8.
    """

    # round predictions to closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() # convert float for division
    acc = correct.sum() / len(correct)

    return acc

# define training loop
epochs = 1000

for epoch in range(epochs):
    # Convert data to PyTorch tensors
    inputs = torch.Tensor(X)
    # reshapes labels from single row to single column
    labels = torch.Tensor(y).reshape(-1, 1)

    # First zero the gradients
    optimizer.zero_grad()

    # Forward pass

    # Make predictions from model
    outputs = model(inputs)
    # calculate loss, "how far off is the prediction from the true label?"
    loss = criterion(outputs, labels)
    # calculate accuracy, "among all the examples, how many are correctly predicted?"
    acc = binary_accuracy(outputs, labels)

    # Backward pass and optimization

    # Calculate the gradient 
    loss.backward()
    # Update the model's parameters using the gradient and optimizer algo
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f},\
               Accuracy: {acc.item()*100:.2f}%")


