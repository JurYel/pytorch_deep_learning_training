num1 = int(input("Enter num1: "))
num2 = int(input("Enter num2: "))

def add_nums(a, b):
    return a + b

def subtract_nums(a, b):
    return a - b

def multiply_nums(a, b):
    return a * b

def divide_nums(a, b):
    return a / b

result1 = add_nums(num1, num2)
result2 = subtract_nums(num1, num2)
result3 = multiply_nums(num1, num2)
result4 = divide_nums(num1, num2)

print(f"{num1} + {num2} = {result1}")
print("{0} - {1} = {2}".format(num1, num2, result2))
print(f"{num1} * {num2} = {result3}")
print(f"{num1} / {num2} = {result4}")