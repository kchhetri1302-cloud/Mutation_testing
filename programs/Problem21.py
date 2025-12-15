#Insert an element at a given position into a list

def insertElementAt(elementToBeInsert, myList, elementNr):
    return myList[:elementNr-1]+[elementToBeInsert]+myList[elementNr-1:]

myList = [0, 1, 2, 3, 4, 5, 6]
print(insertElementAt(5, myList, 3))
