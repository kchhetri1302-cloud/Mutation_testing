def greatestCommonDivisor(a, b):

	if (a == 0):
		return b
	return greatestCommonDivisor(b % a, a)

def TotientFunction(n):

	result = 1
	for i in range(2, n):
		if (greatestCommonDivisor(i, n) == 1):
			result+=1
	return result

for n in range(1, 11):
	print("TotientFunction(",n,") = ", 
		TotientFunction(n), sep = "")
