import itertools

#Determine whether a given integer number is prime

num = int(input("Enter an integer: "))

counter = 0

for i in range(1, num+1):
    if (num % i) == 0:
        counter += 1
        if counter > 2:
            print("The number is not prime")
            break

if counter == 2:
    print("The number is prime")