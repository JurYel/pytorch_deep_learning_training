class Car:
    # constructor method

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    # Method to get the car's age
    def get_age(self):
        return 2023 - self.year
    

my_car = Car(make="Ford", model="Mustang", year=2012)

print(my_car.make)
print(my_car.model)
print(my_car.year)

print(my_car.get_age())