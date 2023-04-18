import torch

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([10, 20])

z = x + y  # broadcasting operation
print(z) 
# tensor([[11, 22],
#        [13, 24]])