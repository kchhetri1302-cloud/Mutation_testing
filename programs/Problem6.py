myList = []
x = int(input("Number of elements: "))
print("Now insert the elements: ")
for i in range(0,x):
    elements = int(input())
    myList.append(elements)
print(myList)

if myList == myList[::-1]:
    print("It's a palindrome")
else:
    print("It's not a palindrome")