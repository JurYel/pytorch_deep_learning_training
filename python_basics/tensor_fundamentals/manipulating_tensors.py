import torch

# reshape with .view()

# create a tensor of shape (2,3)
x = torch.tensor([[1,2,3], [4,5,6,]])

# reshape the tensor to shape (3,2)
y = x.view(3,2)

print(x.shape) # torch.Size([2, 3])
print(x)
# output: tensor([[1, 2, 3],
#                 [4, 5, 6]])

print(y.shape) # torch.Size([3, 2])
print(y) 
# output: tensor([[1, 2],
#                 [3, 4],
#                 [5, 6]])

# =====================
import torch

# reshape with .reshape()

# create a tensor of shape (3, 4)
x = torch.arange(1, 9)

# reshape to (6,2)
y = x.reshape(2, 4)

print(x.shape) # torch.Size([8])
print(x) 
# output: tensor([1, 2, 3, 4, 5, 6, 7, 8])

print(y.shape) # torch.Size([2, 4])
print(y)
# output: tensor([[1, 2, 3, 4],
#                 [5, 6, 7, 8]])

# --------------------------------------
# create a 2x3 tensor
x = torch.tensor([[1,2,3], [4,5,6]])

# use view to reshape the tensor to a 3x2 tensor
y = x.view(3,2)

# use reshape to create a 1D tensor
z = x.reshape(6)

print(x)
# output: tensor([[1,2,3],
#                 [4,5,6]])

print(y)
# output: tensor([[1,2],
#                 [3,4],
#                 [5,6]])

print(z)
# output: tensor([1,2,3,4,5,6])

#--------------
# concatenate tensors with .cat()

# create two 2x3 tensors
x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor([[7,8,9], [10,11,12]])

# concatenate the tensors along the rows (axis 0)
z = torch.cat((x,y), dim=0)

print(z)
# output: tensor([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9],
#                 [10,11,12]])

# -------------------
# splitting tensors with .split()

# create a 3x6 tensor
x = torch.tensor([[1, 2, 3, 4, 5, 6], 
                  [7, 8, 9, 10, 11, 12], 
                  [13, 14, 15, 16, 17, 18]])

# split the tensor into three 1x6 tensors along axis 0
y = torch.split(x, 1, dim=0)

print(y)
# output: (tensor([[1, 2, 3, 4, 5, 6]]), 
#          tensor([[ 7,  8,  9, 10, 11, 12]]), 
#          tensor([[13, 14, 15, 16, 17, 18]]))

# -------------
# slicing tensors

# create a 3x3 tensor
x = torch.tensor([[12,3,4], 
                 [5,1,2], 
                 [8,2,3]])
print(x[0, :2]) # output: tensor([12, 3])

# ---------------

# transposing tensors

x = torch.tensor([[1,2], [3,4]])
print(x.t()) # output: tensor([[1,3], [2,4]])

# -------------------
# converting tensors to numpy array

import torch
import numpy as np

x = torch.tensor([[1,2], [3,4]])
y = x.numpy()

print(type(y)) # output: <class 'numpy.ndarray'>

