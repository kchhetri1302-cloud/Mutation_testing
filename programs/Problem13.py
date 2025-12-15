from itertools import groupby

def directEncode(myList):
    def secondFunction(keyItem, groupItem):
        length = len(list(groupItem))
        if length > 1: return [length, keyItem]
        else: return keyItem
    return [secondFunction(key, group) for key, group in groupby(myList)]

myList = [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 9]
print(directEncode(myList))
