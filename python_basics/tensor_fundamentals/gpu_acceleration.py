import torch

# moving tensors to gpu

x = torch.rand(3, 3)

# move to gpu with .to()
x = x.to('cuda')

print(x)
# output: tensor([[0.1755, 0.5552, 0.3704],
#                 [0.7541, 0.3339, 0.4275],
#                 [0.1858, 0.5571, 0.3280]], 
#                 device='cuda:0')

# -----------------
# move back to cpu

import torch

x = torch.rand(3,3)

# move to cpu
x = x.to('cpu')

print(x)
# output: tensor([[0.1755, 0.5552, 0.3704],
#                 [0.7541, 0.3339, 0.4275],
#                 [0.1858, 0.5571, 0.3280]])


# ---------------
import torch

# move large tensors to gpu

x = torch.rand(10000, 10000)
y = torch.rand(10000, 10000)

# speed up computation by moving to gpu
x = x.to('cuda')
y = y.to('cuda')

z = x + y

print(z)