def decodeFunction(aList):
    def second_fcn(g):
        if isinstance(g, list): return [(g[1], range(g[0]))]
        else: return [(g, [0])]
    return [x for g in aList for x, R in second_fcn(g) for i in R]

aList = [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 9]
print(decodeFunction(aList))