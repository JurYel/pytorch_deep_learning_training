class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("The animal makes a sound.")

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)

    def make_sound(self):
        print("The dog barks")

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name)

    def make_sound(self):
        print("The cat meows")

class Fish(Animal):
    def __init__(self, name):
        super().__init__(name)

# Create instances of each animal
dog = Dog("Rex")
cat = Cat("Whiskers")
fish = Fish("Nemo")

# Call the make_sound method for each animal
dog.make_sound()
cat.make_sound()
fish.make_sound()