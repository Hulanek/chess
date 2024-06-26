import chess
import numpy as np

import Functions

searchDepth = 3
bestMove = list()
bestEvals = list()
branch = 0

def eval(fen):

    return Functions.fen_eval_stockfish(fen)

def alphaBeta(board, depth, alpha, beta, maximize):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000
        else:
            return 10000
    if depth == 0:
        return eval(board.fen())
    legals = board.legal_moves
    if(maximize):
        bestVal = -99999
        for move in legals:
            board.push(move)
            newEval = alphaBeta(board, depth - 1, alpha, beta, (not maximize))
            if bestVal < newEval:
                bestVal = newEval
                if depth == searchDepth:
                    bestMove.append(move)
                    bestEvals.append(bestVal)
            board.pop()
            alpha = max(alpha, bestVal)
            if alpha >= beta:
                return bestVal
        return bestVal
    else:
        bestVal = 99999
        for move in legals:
            board.push(move)
            newEval = alphaBeta(board, depth - 1, alpha, beta, (not maximize))
            if bestVal > newEval:
                bestVal = newEval
                if depth == searchDepth:
                    bestMove.append(move)
                    bestEvals.append(bestVal)
            board.pop()
            beta = min(beta, bestVal)
            if beta <= alpha:
                return bestVal
        return bestVal


board = chess.Board("8/5pk1/6p1/8/7p/8/5KPP/8 w - - 3 3")
for i in range(5):
    print('-----' + str(i) + '-----')
    alphaBeta(board, searchDepth, -9999, 9999, board.turn)
    print(bestMove)
    print(bestEvals)
    board.push(bestMove[-1])
    bestMove.clear()
    bestEvals.clear()

