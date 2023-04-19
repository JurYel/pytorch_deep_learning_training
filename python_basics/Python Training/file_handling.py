# writing to a file
file = open("myfile.txt", "w")

fruits = ["durian", "peach", "orange", "guava", "dragonfruit"]

for fruit in fruits:
    file.write(fruit + "\n")

file.close()

file2 = open("myfile.txt", "r")
count = 0

while True:

    # get line
    line = file2.readline()

    # check end of file (eof)
    if not line:
        break

    print(f"Line-{count}: {line.strip()}")
    count += 1

file2.close()




