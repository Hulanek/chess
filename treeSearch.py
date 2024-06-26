import chess
import numpy as np
import Functions

def eval(fen):
    return Functions.fen_eval_stockfish(fen)

def alphaBeta(board, depth, alpha, beta, maximize, move_sequence):
    fen = board.fen()

    #if fen in transposition_table:
        #if transposition_table[fen]['depth'] >= depth:
            #return transposition_table[fen]['value'], move_sequence

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000, move_sequence
        else:
            return 10000, move_sequence
    if depth == 0:
        return eval(board.fen()), move_sequence

    if maximize:
        bestVal = -99999
        bestSequence = move_sequence
        legals = board.legal_moves
        for move in legals:
            board.push(move)
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move])
            if bestVal < newEval:
                bestVal = newEval
                bestSequence = newSequence
                print(f'move {move}: {bestVal} and depth: {depth}')
            board.pop()
            alpha = max(alpha, bestVal)
            if alpha >= beta:
                return bestVal, bestSequence
        return bestVal, bestSequence
    else:
        bestVal = 99999
        bestSequence = move_sequence
        legals = board.legal_moves
        for move in legals:
            board.push(move)
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move])
            if bestVal > newEval:
                bestVal = newEval
                bestSequence = newSequence
                print(f'move {move}: {bestVal} and depth: {depth}')
            board.pop()
            beta = min(beta, bestVal)
            if beta <= alpha:
                return bestVal, bestSequence
        return bestVal, bestSequence

boardTest = chess.Board("8/8/5ppp/8/1k3PPP/8/8/1K6 w - - 2 2")
print("pici fen", boardTest.board_fen())
print(eval(boardTest.fen()))
print(eval("8/8/5ppp/8/1k3PPP/8/8/1K6 w - - 2 2"))


board = chess.Board("3rrbk1/1p3ppp/1qp1p3/p2nN3/3PR2P/PPQ5/1BP2PP1/4R1K1 w - - 3 23")


transposition_table = {}
bestVal, bestSequence = alphaBeta(board, 3, -99999, 99999, (board.turn == chess.WHITE), [])

print(bestSequence)
for move in bestSequence:
    print(move)



