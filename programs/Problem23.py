#Extract a given number of randomly selected elements from a list

import random
def randomSelection(myList, numberOfElements):
    return random.sample(myList, numberOfElements)

myList = [0, 1, 2, 3, 4, 5, 6]
print(randomSelection(myList, 5))
