age = int(input("Enter your age: "))

if age < 18:
    print("Sorry, you are not eligible to vote.")
elif age >= 18 and age <= 120:
    print("You are eligible to vote.")
else:
    print("You are no longer required to vote.")