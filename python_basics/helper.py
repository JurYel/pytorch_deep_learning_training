class Car:
    # Constructor method

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    # Method to get the car's age
    def get_age(self):
        return 2023 - self.year
    

my_car = Car("Ford", "Mustang", 2010)

print(my_car.make) # Output: "Ford"
print(my_car.get_age()) # Output: 13