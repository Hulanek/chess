import chess
import chess.polyglot
import sys

size = 64 # has to be power of two 2^x

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
        print("Info about Transposition table")
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
        tempEntry = TEntry()
        tempEntry = self.TTArray[zobrist & (size - 1)]
        isSameZobrist = (tempEntry.zobrist_hash == zobrist)
        return tempEntry, isSameZobrist

board = chess.Board()
legals = board.legal_moves
zobrist_hash = chess.polyglot.zobrist_hash(board)
evaluation = 0.321500
for move in legals:
    oneMove = move


TTinstance = TT()
TTinstance.writeEntry(zobrist_hash, evaluation, oneMove, 5, False, 6)
Entry, isSameZobrist = TTinstance.readEntry(zobrist_hash)
board = chess.Board()
zobrist_hash = chess.polyglot.zobrist_hash(board)
TTinstance.writeEntry(zobrist_hash, evaluation, oneMove, 6, False, 6)
print(Entry.zobrist_hash, Entry.evaluation)
print(isSameZobrist)
print(TTinstance.contentOfTT())
