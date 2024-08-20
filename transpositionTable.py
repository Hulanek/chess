import chess
import chess.polyglot
import sys

size = 65536 # has to be power of two 2^x

class TEntry:
    zobrist_hash = 0
    evaluation = 0
    best_move = chess.Move(0, 0)
    depth = 0
    flag = False
    move_number = 0


class TT:
    def __init__(self):
        self.TTArray = []
        for i in range(size):
            self.TTArray.append(TEntry())


    def infoTT(self):
        usedEntries = 0
        numDepthZero = 0
        numExact = 0
        numUpper = 0
        numLower = 0

        percentageCover = 0

        for i in range(len(self.TTArray)):
            if self.TTArray[i].zobrist_hash != 0:
                usedEntries += 1
                if self.TTArray[i].depth == 0:
                    numDepthZero += 1

        percentageCover = usedEntries / len(self.TTArray) * 100
        print("Info about Transposition table")
        print("Num of used Entries:", usedEntries)
        print("Num of depth zero:", numDepthZero)
        print("Percentage cover:", percentageCover)




    def contentOfTT(self):
        for i in range(len(self.TTArray)):
            print(self.TTArray[i].zobrist_hash, self.TTArray[i].evaluation, self.TTArray[i].best_move, self.TTArray[i].depth, self.TTArray[i].flag, self.TTArray[i].move_number)

    def writeEntry(self, zobrist_hash, evaluation, best_move, depth, flag, move_number):
        tempEntry = TEntry()
        tempEntry.zobrist_hash = zobrist_hash
        tempEntry.evaluation = evaluation
        tempEntry.best_move = best_move
        tempEntry.depth = depth
        tempEntry.flag = flag
        tempEntry.move_number = move_number
        indexHash = zobrist_hash & (size - 1)
        self.TTArray[indexHash] = tempEntry

    def readEntry(self, zobrist):
        tempEntry = self.TTArray[zobrist & (size - 1)]
        isSameZobrist = (tempEntry.zobrist_hash == zobrist)
        return tempEntry, isSameZobrist
