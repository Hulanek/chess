import chess
import numpy as np
import Functions
import keras
from chess import pgn
model = keras.saving.load_model('firstModel.keras')


def eval(fen):
    bitboard = Functions.fen_to_bitboards(fen)
    bitboard = bitboard.reshape(1, 15, 8, 8)
    return model.predict(bitboard, verbose=0)


def alphaBeta(board, depth, alpha, beta, maximize, move_sequence):
    fen = board.fen()

    #if fen in transposition_table:
        #if transposition_table[fen]['depth'] >= depth:
            #return transposition_table[fen]['value'], move_sequence


def alphaBeta(board, depth, alpha, beta, maximize, move_sequence):
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
            board.pop()
            beta = min(beta, bestVal)
            if beta <= alpha:
                return bestVal, bestSequence
        return bestVal, bestSequence

#board = chess.Board("8/5pk1/6p1/8/7p/8/5KPP/8 w - - 3 3")
#for i in range(5):
 #   print('-----' + str(i) + '-----')
  #  alphaBeta(board, searchDepth, -9999, 9999, board.turn, [])
   # print(bestMove)
    #print(bestEvals)
    #board.push(bestMove[-1])
    #bestMove.clear()
    #bestEvals.clear()


#board = chess.Board("3rrbk1/1p3ppp/1qp1p3/p2nN3/3PR2P/PPQ5/1BP2PP1/4R1K1 w - - 3 23")


#transposition_table = {}
#bestVal, bestSequence = alphaBeta(board, 3, -99999, 99999, (board.turn == chess.WHITE), [])

#print(bestSequence)
#for move in bestSequence:
 #   print(move)

board = chess.Board()
while not board.is_checkmate():
    if board.is_checkmate() or board.is_stalemate():
        if board.turn == chess.WHITE:
            print('WHITE LOST')
        else:
            print('BLACK LOST')
    else :
        bestVal, bestSequence = alphaBeta(board, 3, -99999, 99999, board.turn, [])
        board.push(bestSequence[0])
        print(f"{bestSequence[0]}I JUST MADE THE BEST FUCKING MOVE EVER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

