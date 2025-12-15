from itertools import groupby

def modifiedEncode(myList):
        def secondFunction(length):
            if len(length) > 1: 
                return [len(length), length[0]]
            else: return length[0]
        return [secondFunction(list(group)) for key, group in groupby(myList)]

aList = [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 9]
print(modifiedEncode(aList))