#Split a list into two parts; the length of the first part is given

def splitList(myList, N):
    return myList[:N], myList[N:]

myList = [0, 1, 2, 3, 4, 5, 6]
print(splitList(myList, 3))