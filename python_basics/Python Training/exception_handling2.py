try:
    file = open("example2.txt", "r")
    # Do something with the file
    # print(file.read())
except IOError:
    print("An error occured while reading the file.")
finally:
    file.close()