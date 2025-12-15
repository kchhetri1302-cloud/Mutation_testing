#Drop every N'th element from a list

def dropElement (myList, N):
    return [item for i, item in enumerate(myList) if (i+1) % N]

myList = [0, 1, 2, 3, 4, 5, 6]
print(dropElement(myList, 3))