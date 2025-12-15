myList = []
x = int(input("Number of elements: "))
print("Now insert the elements: ")
for i in range(0,x):
    elements = int(input())
    myList.append(elements)
print(myList)

y = int(input("List index: "))

Kthelement = myList[y-1]

print("The element with index " + str(y) + " is: ", Kthelement)
