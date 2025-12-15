#Generate the combinations of K distinct objects chosen from the N elements of a list

import itertools

def combinationsFunction (List, nElements):
    if (len(List) >= nElements): 
        combinations = list(itertools.combinations(List, nElements))
    else:
        print("N must be smaller than the length of the list")
    return combinations

myList = [1, 2, 3, 4, 5, 6, 7, 8]
print(combinationsFunction(myList, 2))
