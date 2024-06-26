import chess
import numpy as np

import Functions


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
                bestLayerMove = move
                print("new best move found for White", depth, bestLayerMove, bestVal)
            #print("depth", depth, "move", move, "bestVal", bestVal)
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
                bestLayerMove = move
                print("new best move found for Black", depth, bestLayerMove, bestVal)
            board.pop()
            beta = min(beta, bestVal)
            if beta <= alpha:
                return bestVal
        return bestVal


boardTest = chess.Board("8/8/5ppp/8/1k3PPP/8/8/1K6 w - - 2 2")
print("pici fen", boardTest.board_fen())
print(eval(boardTest.fen()))
print(eval("8/8/5ppp/8/1k3PPP/8/8/1K6 w - - 2 2"))


board = chess.Board("8/8/5ppp/8/1k3PPP/8/8/1K6 w - - 2 2")
alphaBeta(board, 3, -9999, 9999, True)
