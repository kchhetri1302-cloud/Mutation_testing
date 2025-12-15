# Determine if two numbers are coprime

a = int(input("First number "))
b = int(input("Second number ")) 
remainder=a%b
while remainder:
    a = b
    b = remainder
    remainder = a%b
#print('GCD is:', b)

if (b != 1):
    print("The numbers are not coprime")
else:
    print("The numbers are coprime")