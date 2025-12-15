import itertools

#Sorting a list of lists according to length of sublists

inputList = [[1], [5, 5, 5], [7, 7], [5, 5, 5], [10, 10, 10]]

inputList.sort(key=len)

print(inputList)

#it prints [[1], [7, 7], [5, 5, 5], [5, 5, 5], [10, 10, 10]]