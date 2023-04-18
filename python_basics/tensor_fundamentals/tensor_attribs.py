import torch

# create a tensor
tensor = torch.tensor([[2,4,6],
                       [5,10,15],
                       [9,18,27]])

# get the rank or number of dimensions
print(tensor.ndim)

# get the shape of the tensor
print(tensor.shape)

# get the data type of a a tensor
print(tensor.dtype)

# get the device the tensor is stored
print(tensor.device)