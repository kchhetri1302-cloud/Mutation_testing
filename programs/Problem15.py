#Duplicate the elements of a list a given number of times

def duplicate(myList, N):
    return [item for item in myList for i in range(N)]

myList = [0, 1, 2, 3, 4, 5, 6]
print(duplicate(myList, 3))