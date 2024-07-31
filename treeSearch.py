import random
import chess
import ordered_moving
from onnxruntime import InferenceSession
import Functions
import time
import threading
#model = keras.models.load_model('firstModel.keras')

sess = InferenceSession('2conv32_33_3dense_768_256.onnx')
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



board = chess.Board()
timeLeft = 100

turnTime = timeLeft / 25  # allocate 1/25th part of time left for current turn

for i in range(1, 6):
    depth_time_start = time.time()
    # asi ukladat posledni sekvenci -> ta se asi vyzkousi jako prvni pri dalsi iteraci
    bestVal, bestSequence = alphaBeta(board, i, -99999, 99999, board.turn, [])
    print("info depth", i, "score cp", bestVal[0] * 1000, "pv", bestSequence)

    depth_time_end = time.time()
    depth_duration = depth_time_end - depth_time_start

    turnTime = turnTime - (depth_duration)
    print(depth_duration, turnTime)


while True:
    args = input().split()

    if args[0] == "uci":
        print("model_name")
        print("uciok")

    elif args[0] == "isready":
        print("readyok")

    elif args[0] == "quit":
        break

    # podle tech oficialnich pravidel by to melo jit nastavit podle fenu ale to zatim nemame
    elif args[:2] == ["position", "startpos"]:
        board = chess.Board()
        for ply, move in enumerate(args[3:]): # od indexu 4 vynecha slovo moves
            board.push(chess.Move.from_uci(move))


    elif args[0] == "go":
        bestVal, bestSequence = alphaBeta(board, 5, -99999, 99999, board.turn, [])
        print("bestmove", bestSequence[0])
