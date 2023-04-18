import torch




# Sum on specific dims

x = torch.tensor([[1, 2], [3, 4]])

# sum of all elements in tensor
z = x.sum()  
print(z) # tensor(10)

# sum of elements along the first dimension
z = x.sum(dim=0)  
print(z) # tensor([4, 6])

# sum of elements along the second dimension
z = x.sum(dim=1)  
print(z) # tensor([3, 7])


# -------------------
# Sum
# Define a tensor
a = torch.tensor([[1, 2], [3, 4]])

# Find the sum of all values in the tensor
b = torch.sum(a)

print(b) # tensor(10)

# -------------------
# Mean

# Define a tensor
a = torch.tensor([[1, 2], [3, 4]], \
                 dtype=torch.float32)

# Find the mean of all elements in the tensor
b = torch.mean(a)

print(b) # tensor(2.5000, dtype=torch.float32)

# -----------------------------

# Mean on specific dims

x = torch.tensor([[1, 2], [3, 4]], \
                 dtype=torch.float32)

# mean of all elements in tensor
z = x.mean()
print(z) # tensor(2.5000)

# mean of elements along the first dimension
# calculates mean across columns then groups them
z = x.mean(dim=0)  
print(z) # tensor([2., 3.])

# mean of elements along the second dimension
# calculates mean in each vector then groups them
z = x.mean(dim=1)  
print(z) # tensor([1.5000, 3.5000])

# ---------------------------
# Max and Min

# Maximum on specific dims

x = torch.tensor([[1, 2], [3, 4]], \
                 dtype=torch.float32)

# maximum of all elements in tensor
z = x.max()
print(z) # tensor(4.)

# maximum of elements along the first dimension
# returns the vector with highest values
z = x.max(dim=0)  
print(z) # torch.return_types.max(
         # values=tensor([ 3., 4.]),
         # indices=tensor([1, 1]))

# minimum of elements along the second dimension
# returns the lowest value in each vector and groups them
z = x.max(dim=1)
print(z) # torch.return_types.max(
         # values=tensor([2., 4.]),
         # indices=tensor([1, 1]))
#------------------------

# Minimum on specific dims

x = torch.tensor([[1,2], [3,4]], \
                 dtype=torch.float32)

# minimum of all elements in tensor
z = x.min()
print(z) # tensor(1.)

# minimum of elements along the first dimension
# returns the vector with lowest values
z = x.min(dim=0)
print(z) # torch.return_types.min(
         # values=tensor([1., 2.]),
         # indices=tensor([0, 0]))

# minimum of elements along the second dimension
# returns the lowest value in each vector and groups them
z = x.min(dim=1)
print(z) # torch.return_types.min(
         # values=tensor([1.,3.]),
         # indices=tensor([0, 0]))