import random
import chess
import numpy as np
import keras
from chess import pgn
import time
import tensorflow as tf
from onnxruntime import InferenceSession
import Functions
#model = keras.models.load_model('firstModel.keras')

sess = InferenceSession('new_bitboards_model.onnx')
num_of_nodes = 0
def randomPlay(board):
    legals = list(board.legal_moves)
    return legals[random.randint(0, len(legals)-1)]

def eval(board):
    global num_of_nodes
    num_of_nodes = num_of_nodes + 1
    bitboards = Functions.boardToBitboard(board)
    bitboards = bitboards.reshape(1, 15, 8, 8)
    return sess.run(None, {'input': bitboards})[0]

def alphaBeta(board, depth, alpha, beta, maximize, move_sequence):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000, move_sequence
        else:
            return 10000, move_sequence
    if depth == 0:
        return eval(board), move_sequence

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


# game settings
game = chess.pgn.Game()
game.headers["White"] = "White player name"
game.headers["Black"] = "Black player name"
game.headers["Event"] = "Ultra prestige turnament at Zlín and Prague"

board = chess.Board()

node = game


while not board.is_game_over():
#for i in range(10):
    if board.turn == chess.WHITE:
        move = randomPlay(board)
        board.push(move)
        node = node.add_variation(move)
        print(f"{move}")
    else:
        num_of_nodes = 0
        start = time.time()
        bestVal, bestSequence = alphaBeta(board, 6, -99999, 99999, board.turn, [])
        board.push(bestSequence[0])
        node = node.add_variation(bestSequence[0])
        end = time.time()
        print(f"{bestSequence[0]}", "num of checked nodes - ", num_of_nodes, "time of tree search - ", end - start)

game.headers["Result"] = board.result()
print(game)
