N = 8 # 8 x 8 - size of the chessboard

def EightQueens(chessBoard, boardColumn):
	if boardColumn == N:
		print(chessBoard)
		return True
	for i in range(N):
		if notIntersecting(chessBoard, i, boardColumn):
			chessBoard[i][boardColumn] = 1
			if EightQueens(chessBoard, boardColumn + 1):
				return True
			chessBoard[i][boardColumn] = 0
	return False

def notIntersecting(chessBoard, boardRow, boardColumn):
	for i in range(boardColumn):
		if chessBoard[boardRow][i] == 1:
			return False
	for i, j in zip(range(boardRow, -1, -1), range(boardColumn, -1, -1)):
		if chessBoard[i][j] == 1:
			return False
	for i, j in zip(range(boardRow, N, 1), range(boardColumn, -1, -1)):
		if chessBoard[i][j] == 1:
			return False
	return True

chessBoard = [[0 for i in range(N)] for j in range(N)]
if not EightQueens(chessBoard, 0):
	print("There is no solution")
