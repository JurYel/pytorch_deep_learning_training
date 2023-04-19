class Calculator:
    def add(self, a, b=0, c=0):
        return a + b + c

calc = Calculator()
print(calc.add(2))
print(calc.add(2, 100))
print(calc.add(2, 100, 1000))