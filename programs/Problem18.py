#Extract a slice from a list

def sliceList(myList, fromHere, toHere):
    return myList[fromHere-1:toHere]

myList = [0, 1, 2, 3, 4, 5, 6]
print(sliceList(myList, 3, 6))