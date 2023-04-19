try:
    num = int(input("Enter a number: "))
    print("The reciprocal of ", num, " is ", 1/num)
except ValueError:
    print("Please enter a valid integer.")
except ZeroDivisionError:
    print("Cannot divide by zero.")