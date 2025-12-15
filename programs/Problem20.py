def removeKthElement(myList, elementNr):
    return myList[elementNr-1], myList[:elementNr-1] + myList[elementNr:]

myList = [0, 1, 2, 3, 4, 5, 6]
print(removeKthElement(myList, 3))