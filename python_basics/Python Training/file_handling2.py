# append to file
file = open("myfile.txt", "a")

vegetables = ["malunggay\n", "talong\n", "luya\n", "bawang\n", "sibuyas\n"]

for veggie in vegetables:
    file.write(veggie)

file.close()

# read the file
file1 = open('myfile.txt', "r")
count = 0

while True:
    
    # get the line
    line = file1.readline()

    # check eof
    if not line:
        break
    
    count += 1
    print("Line-{}: {}".format(count, line))

file1.close()

names = ["Alex", "Jonah", "Messi", "Eren", "Ronaldo"]

with open("myfile.txt", "w") as f:
    for name in names:
        f.write(name + "\n")

with open("myfile.txt", "r") as f:
    print(f.read())