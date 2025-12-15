#Generate a random permutation of the elements of a list

import random
def randomPermutation(L):
    return random.sample(myList, len(myList))

myList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(randomPermutation(myList))
