#Create a list containing all integers within a given range

def inrange(i, j):
    return range(i, j+1)

x = []
x = inrange(2, 7)

for i in x:
    print (i)
