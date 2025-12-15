myList = []
x = int(input("Number of elements: "))
print("Now insert the elements: ")
for i in range(0,x):
    elements = int(input())
    myList.append(elements)
print(myList)

penultimate = myList[-2]
print("The last but one element is: ", penultimate)
