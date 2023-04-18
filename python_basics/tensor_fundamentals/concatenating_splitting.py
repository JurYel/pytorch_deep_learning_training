import torch

# concatenating tensors with .cat()

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
z = torch.cat((x,y), dim=0)
z2 = torch.cat((x,y), dim=1)

print(z)
# output: tensor([[1, 2],
#                 [3, 4],
#                 [5, 6],
#                 [7, 8]])

print(z2)
# output: tensor([[1, 2, 5, 6],
#                 [3, 4, 7, 8]])

# ---------- 

import torch

# concatenating tensors with .stack()

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = torch.tensor([5, 6])
d = torch.stack((a, b, c))

print(d)
# output: tensor([[1, 2],
#                 [3, 4],
#                 [5, 6]])

# ------------------

import torch

# splitting with .split()

x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
y1, y2 = torch.split(x, 2, dim=0)
y3, y4 = torch.split(x, 2, dim=1)

print(y1)
# output: tensor([[1, 2, 3],
#                 [4, 5, 6]])

print(y2)
# output: tensor([[7, 8, 9],
#                 [10, 11, 12]])

print(y3)
# output: tensor([[ 1,  2],
#                 [ 4,  5],
#                 [ 7,  8],
#                 [10, 11]])

print(y4)
# output: tensor([[ 3],
#                 [ 6],
#                 [ 9],
#                 [12]])

# -------------------------
import torch

# chunking with .chunk()

x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

z1, z2, z3 = torch.chunk(x, 3, dim=1)

print(z1)
# output: tensor([[1],
#                 [4],
#                 [7],
#                 [10]])

print(z2)
# output: tensor([[2],
#                 [5],
#                 [8],
#                 [11]])

print(z3)
# output: tensor([[3],
#                 [6],
#                 [9],
#                 [12]])