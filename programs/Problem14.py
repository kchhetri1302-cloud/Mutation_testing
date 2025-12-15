def duplicateItems(myList):
    return [element for element in myList for i in (1,2)]

myList = [0, 1, 2, 3, 4, 5, 6]
print(duplicateItems(myList))
