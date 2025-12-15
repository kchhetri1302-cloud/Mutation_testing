gridSize = 9
def printMatrix(arr):
	for i in range(gridSize):
		for j in range(gridSize):
			print(arr[i][j], end = " ")
		print()

def safeToPopulate(grid, row, column, num):
	for x in range(9):
		if grid[row][x] == num:
			return False

	for x in range(9):
		if grid[x][column] == num:
			return False

	startRow = row - row % 3
	startcolumn = column - column % 3
	for i in range(3):
		for j in range(3):
			if grid[i + startRow][j + startcolumn] == num:
				return False
	return True 

def solveSudokuGrid(grid, row, column):
	if (row == gridSize - 1 and column == gridSize):
		return True

	if column == gridSize:
		row += 1
		column = 0

	if grid[row][column] > 0:
		return solveSudokuGrid(grid, row, column + 1)
	for num in range(1, gridSize + 1, 1):

		if safeToPopulate(grid, row, column, num):

			grid[row][column] = num

			if solveSudokuGrid(grid, row, column + 1):
				return True

		grid[row][column] = 0
	return False

grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
		[5, 2, 0, 0, 0, 0, 0, 0, 0],
		[0, 8, 7, 0, 0, 0, 0, 3, 1],
		[0, 0, 3, 0, 1, 0, 0, 8, 0],
		[9, 0, 0, 8, 6, 3, 0, 0, 5],
		[0, 5, 0, 0, 9, 0, 6, 0, 0],
		[1, 3, 0, 0, 0, 0, 2, 5, 0],
		[0, 0, 0, 0, 0, 0, 0, 7, 4],
		[0, 0, 5, 2, 0, 6, 3, 0, 0]]

if (solveSudokuGrid(grid, 0, 0)):
	printMatrix(grid)
else:
	print("No solutions")
