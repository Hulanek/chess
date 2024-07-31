import random
import chess
import numpy as np
import keras
from chess import pgn
import time
import ordered_moving
from onnxruntime import InferenceSession
import Functions
#model = keras.models.load_model('firstModel.keras')

sess = InferenceSession('4dense_2dropouts02_10k.onnx')
num_of_nodes = 0
sum_of_nodes = 0

def randomPlay(board):
    legals = list(board.legal_moves)
    return legals[random.randint(0, len(legals)-1)]

def eval(board):
    global num_of_nodes
    num_of_nodes = num_of_nodes + 1
    bitboards = Functions.boardToBitboard(board)
    bitboards = bitboards.reshape(1, 13, 8, 8)
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
        bestVal = -99999
        bestSequence = move_sequence
        legals = board.legal_moves
        ordered_moves = ordered_moving.move_ordering(legals, board)
        ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)
        for move in ordered_moves:
            board.push(move[0])
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move[0]])
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
        ordered_moves = ordered_moving.move_ordering(legals, board)
        ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)
        for move in ordered_moves:
            board.push(move[0])
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move[0]])
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



for i in range(10):
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
            bestVal, bestSequence = alphaBeta(board, 2, -99999, 99999, board.turn, [])
            if bestSequence:
                board.push(bestSequence[0])
                node = node.add_variation(bestSequence[0])
                end = time.time()
                print(f"{bestSequence[0]}", "num of checked nodes - ", num_of_nodes, "time of tree search - ", end - start)
            else:
                print(f"No valid moves found by alphaBeta or maybe just one? {[x for x in board.legal_moves]}.")
                break
                # legal = board.legal_moves
                # if board.legal_moves.count() == 1:
                #     for move in legal:
                #         board.push(move)

    game.headers["Result"] = board.result()
    print(game)
