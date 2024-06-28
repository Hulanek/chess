import random
import chess
import numpy as np
import Functions
import keras
from chess import pgn
import time
import tensorflow as tf

model = keras.models.load_model('firstModel.keras')

def randomPlay(board):
    legals = list(board.legal_moves)
    return legals[random.randint(0, len(legals)-1)]

def eval(fen):
    bitboards = Functions.fen_to_bitboards(fen)
    bitboards = bitboards.reshape(1, 15, 8, 8)
    #start = time.time()
    #predict = model(bitboards)
    #end = time.time()
    #print("model(x)", end - start, predict)
    #start = time.time()
    predict = model.predict_on_batch(bitboards)
    #end = time.time()
    #print("batch_predict", end - start, predict)
    return predict
    #return random.randint(-1000, 1000)

def alphaBeta(board, depth, alpha, beta, maximize, move_sequence):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000, move_sequence
        else:
            return 10000, move_sequence
    if depth == 0:
        return eval(board.fen()), move_sequence

    if maximize:
        bestVal = -9999
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
        bestVal = 9999
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


weights = model.layers[0].get_weights()

# game settings
game = chess.pgn.Game()
game.headers["White"] = "White player name"
game.headers["Black"] = "Black player name"
game.headers["Event"] = "Ultra prestige turnament at Zlín and Prague"

board = chess.Board()

node = game


while not board.is_game_over():
    if board.turn == chess.WHITE:
        move = randomPlay(board)
        board.push(move)
        node = node.add_variation(move)
        print(f"{move}")
    else:
        bestVal, bestSequence = alphaBeta(board, 3, -99999, 99999, board.turn, [])
        board.push(bestSequence[0])
        node = node.add_variation(bestSequence[0])
        print(f"{bestSequence[0]}")

game.headers["Result"] = board.result()
print(game)
