import chess
import random

piece_value = {
    'p': -100,
    'r': -500,
    'n': -320,
    'b': -330,
    'q': -900,
    'k': -20000,
    'P': 100,
    'R': 500,
    'N': 320,
    'B': 330,
    'Q': 900,
    'K': 20000,
}

def materialEvaluation(board):
    materialValue = 0
    for i in range(64):
        if board.piece_at(i)  != None:
            pieceSymbol = board.piece_at(i).symbol()
            materialValue += piece_value[pieceSymbol]
    return materialValue + random.randint(-2, 2)



