import torch

# Define two matrices
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Perform matrix multiplication
c = torch.matmul(a, b)
c = a @ b # same operations

# Print the result
print(c)
# tensor([[19, 22],
#        [43, 50]])

