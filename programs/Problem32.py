#the Python default method
# import math

# a = 100
# b = 250

# c = math.gcd(a, b)
# print(c)

a = int(input("First number "))
b = int(input("Second number ")) 
remainder=a%b
while remainder:
    a = b
    b = remainder
    remainder = a%b
print('GCD is:', b)