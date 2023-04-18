import torch

x = torch.tensor([4,3,2])
y = torch.tensor([5,1,2])

z = x + y # element-wise addition
z = torch.add(x, y) # same operation
print(z) # tensor([9, 4, 4])

z = x - y # element-wise subtraction
z = torch.sub(x, y) # same operation\
print(z) # tensor([-1,  2,  0])

z = x * y # element-wise multiplication
z = torch.multiply(x, y) # same operation
print(z) # tensor([20,  3,  4])

z = x / y # element-wise division
z = torch.divide(x, y) # same operation
print(z) # tensor([0.8000, 3.0000, 1.0000])

z = torch.dot(x, y) # dot product operation
z = x @ y # dot product operation
print(z) # tensor(27)