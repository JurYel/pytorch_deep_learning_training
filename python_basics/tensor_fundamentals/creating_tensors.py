import torch
import numpy as np


# basic construction 
scalar = torch.tensor(5)
print(scalar) # tensor(5)

# creating tensor from list
my_list = [1,2,3,4,5]
my_tensor = torch.tensor(my_list)
print(my_tensor) # tensor([1, 2, 3, 4, 5])

# creating tensor from numpy array
my_array = np.array([1,2,3,4,5])
my_tensor = torch.from_numpy(my_array)
print(my_tensor) # tensor([1, 2, 3, 4, 5])

# creating a tensor with all zeros
my_tensor = torch.zeros(2,2)
print(my_tensor)
# tensor([[0., 0.],
#        [0., 0.]])

# creating a tensor with all ones
my_tensor = torch.ones(3,3)
print(my_tensor)
# tensor([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]])

# creating a random tensor
my_tensor = torch.rand(2,3)
print(my_tensor)
# tensor([[0.9550, 0.6176, 0.1351],
#        [0.5591, 0.3762, 0.5076]])

# create an empty 2d tensor
my_tensor = torch.empty(3,3)
print(my_tensor)