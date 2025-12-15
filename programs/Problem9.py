from itertools import groupby

def group_duplicates(list_nums):
    return [list(group) for key, group in groupby(list_nums)]

myList = [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 9]
print("Original list:") 
print(myList)
print("\nAfter packing consecutive duplicates of the said list elements into sublists:")
print(group_duplicates(myList)) 