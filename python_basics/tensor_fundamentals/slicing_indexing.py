import torch

# indexing tensors

# create a tensor
x = torch.tensor([[1,2,3], 
                  [4,5,6], 
                  [7,8,9]])

# select the element at index (1,2)
print(x[1,2])
# output: tensor(6)

# -------------------
# slicing tensors

import torch

# create a tensor 
x = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

# select the first row of the tensor
print(x[0, :])
# output: tensor([1,2,3])

# select the last two columns of the vector
print(x[:, 1:])
# output: tensor([[2, 3],
#                 [5, 6],
#                 [8, 9]])

# ---------------
# advanced indexing

import torch

# create a tensor
x = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

# select elements at indices (0, 0), (1,2), and (2,1)
indices = torch.tensor([[0, 1, 2],
                        [0, 2, 1]])
print(x[indices])
# output: tensor([1, 6, 8])

# -----------------
# boolean indexing

import torch

# create a tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# select elements greater than 5
print(x[x > 5])
# output: tensor([6, 7, 8, 9])