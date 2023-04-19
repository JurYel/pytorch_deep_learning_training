class ParentClass:
    z = 10
    def __init__(self, x):
        self.x = x

    def my_method(self):
        print("This method is from the parent class.")


class ChildClass(ParentClass):

    def __init__(self, x, y):
        # super().__init__(x)
        self.x = super().z
        self.y = y

    def my_method(self):
        print("This method is from the child class.")

parent = ParentClass(100)
child = ChildClass(20, 50)

print(parent.x)
# output: 100

parent.my_method()

print(child.x)

print(child.y)

child.my_method()