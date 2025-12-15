#Lotto: Draw N different random numbers from the set 1

import random
random.seed(42)
def randomSelect(howManyNumbers, endOfInterval):
    return random.sample(range(1, endOfInterval+1), howManyNumbers)

print(randomSelect(4, 100))
