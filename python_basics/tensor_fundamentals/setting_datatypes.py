import torch

# Specifying data type during tensor creation
my_tensor = torch.tensor([1,2,3], \
                         dtype=torch.float32)
print(my_tensor) # tensor([1., 2., 3.])
print(my_tensor.dtype) # torch.float32

# Casting to a different data type
my_tensor = torch.tensor([14,22,11,43], \
                         dtype=torch.int64)
my_tensor = my_tensor.to(torch.float32)
print(my_tensor) # tensor([14., 22., 11., 43.])
print(my_tensor.dtype) # torch.float32

# Setting default data type
torch.set_default_dtype(torch.float64)
my_tensor = torch.tensor([32.,12.,22.,23.])
print(my_tensor) # tensor([32., 12., 22., 23.])
print(my_tensor.dtype) # torch.float64