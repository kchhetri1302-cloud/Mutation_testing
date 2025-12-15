#Group the elements of a set into disjoint subsets.
import itertools
def disjointSubsets(setOfElements, setOfSubsets):
    sumOfElements = 0
    numberOfGroups = len(setOfSubsets)
    x = list()
    for i in range(0, numberOfGroups):
        sumOfElements = sumOfElements + setOfSubsets[i]
    setOfElementsList = list(setOfElements)
    if len(setOfElementsList) == sumOfElements:
        for j in range(0, numberOfGroups):
            subsets = list(itertools.combinations(setOfElementsList, setOfSubsets[j]))
            x.append(subsets)
            subsetsListBackToSet = set(subsets)
            print(subsetsListBackToSet)
            
    else:
        print("The number of elements in set does not match the sum of subsets array elements")
    print(x)
    return subsetsListBackToSet

mySet = {"gigi", "carla", "david", "gutsy"}
subsetLengths = [2, 1, 1]
disjointSubsets(mySet, subsetLengths)
