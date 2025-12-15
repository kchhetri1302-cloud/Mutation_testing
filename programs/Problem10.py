def runLengthEncode(sequence):
  compressed = []
  count = 1
  char = sequence[0]
  for i in range(1,len(sequence)):
    if sequence[i] == char:
      count = count + 1
    else :
      compressed.append([char,count])
      char = sequence[i]
      count = 1
  compressed.append([char,count])
  return compressed
 
sequence = [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 9]
myList = runLengthEncode(sequence)

print(myList)
compressedSequence = ''
 
for item in range(0,len(myList)):
  for items in myList[item]:
    compressedSequence += str(items)
 
print(compressedSequence)