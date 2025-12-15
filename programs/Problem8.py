myList = []
x = int(input("Number of elements: "))
print("Now insert the elements: ")
for i in range(0,x):
    elements = int(input())
    myList.append(elements)
print(myList)

myList = list(set(myList))
print("The list without duplicates: " + str(myList))