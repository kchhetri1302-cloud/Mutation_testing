def rotateList(myList, fromWhereToRotate):
    return myList[fromWhereToRotate:] + myList[:fromWhereToRotate]

myList = [0, 1, 2, 3, 4, 5, 6]
print(rotateList(myList, 3))