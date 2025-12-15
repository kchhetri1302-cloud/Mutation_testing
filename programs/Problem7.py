myList = [1, 2, 3, 4, [1, 2, 3,], 6]
myFlattenList = []

for j in myList:
    if type(j) == list:
        myFlattenList.extend(j)
    else:
        myFlattenList.append(j)

print("My flatten list is: " + str(myFlattenList))