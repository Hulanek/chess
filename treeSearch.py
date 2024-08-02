import random
import chess
import ordered_moving
from onnxruntime import InferenceSession
import Functions
import time
import threading
#model = keras.models.load_model('firstModel.keras')

sess = InferenceSession('2conv32_33_3dense_768_256.onnx')
TIME_NUM_OF_NODES = 20000
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
    return sess.run(None, {'input': bitboards})[0][0][0]

def alphaBeta(board, depth, alpha, beta, maximize, move_sequence, PV):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000, move_sequence
        else:
            return 10000, move_sequence

    if board.is_stalemate():
        return 0, move_sequence

    if depth == 0:
        return eval(board), move_sequence

    if maximize:
        bestVal = -99999
        bestSequence = move_sequence
        legals = list(board.legal_moves)
        # legals jsou nyni list, kvuli dvoji iteraci

        if PV:
            # kontrola hloubky
            pv_move = PV[len(move_sequence)] if len(PV) > len(move_sequence) else None
            if pv_move in legals:
                # prvek pridan do legals
                legals.remove(pv_move)
                legals.insert(0, pv_move)

        ordered_moves = ordered_moving.move_ordering(legals, board)
        ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)
        for move in ordered_moves:
            board.push(move[0])
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move[0]], PV)
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
        legals = list(board.legal_moves)
        # legals jsou nyni list, kvuli dvoji iteraci

        if PV:
            #kontrola hloubky od rootu a porovnani s PV, PV(3) > 2 -> mame pro depth 2 PV[2]
            pv_move = PV[len(move_sequence)] if len(PV) > len(move_sequence) else None
            if pv_move in legals:
                # prvek pridan do legals
                legals.remove(pv_move)
                legals.insert(0, pv_move)

        ordered_moves = ordered_moving.move_ordering(legals, board)
        ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)
        for move in ordered_moves:
            board.push(move[0])
            newEval, newSequence = alphaBeta(board, depth - 1, alpha, beta, (not maximize), move_sequence + [move[0]], PV)
            if bestVal > newEval:
                bestVal = newEval
                bestSequence = newSequence
            board.pop()
            beta = min(beta, bestVal)
            if beta <= alpha:
                return bestVal, bestSequence
        return bestVal, bestSequence


board = chess.Board("3k4/8/8/8/8/8/7R/5K2 w - - 0 1")
PV = []
start = time.time()
for i in range(1, 7):
    depth_time_start = time.time()
    bestVal, bestSequence = alphaBeta(board, i, -99999, 99999, board.turn, [], PV)
    PV = bestSequence
    PV_string = ""
    for j in range(len(PV)):
        PV_string += chess.Move.uci(PV[j]) + " "
    print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "pv", PV_string)
end = time.time()
print("time taken", end - start)
print("bestmove", PV[0])


while True:
    args = input().split()

    if args[0] == "uci":
        print("model_name")
        print("uciok")

    elif args[0] == "isready":
        print("readyok")

    elif args[0] == "quit":
        break

    # example - position startpos moves e2e4 e7e5
    elif args[0] == "position":
        if args[1] == "startpos":
            board = chess.Board()
            for ply, move in enumerate(args[3:]):  # from index 3 - skips string moves
                board.push(chess.Move.from_uci(move))

    # example - position fen 4k3/8/8/8/8/8/7R/5K2 w - - 0 1 moves h2h8
        elif args[1] == "fen":
            fen = ""
            movesIndex = 2
            for part in args[2:]:
                if(part != "moves"):
                    fen += part + " "
                    movesIndex += 1
                else:
                    break
            board = chess.Board(fen)
            for ply, move in enumerate(args[(movesIndex + 1):]):
                board.push(chess.Move.from_uci(move))


    elif args[0] == "go":
        # chybi inkrementovaci cas
        wtime, btime, moves= [int(a) / 1000 for a in args[2::2]]
        if(board.turn == chess.WHITE):
            timeLeft = wtime
        else:
            timeLeft = btime

        turnTime = timeLeft / 25  # allocate 1/25th part of time left for current turn

        turnTime_start = time.time()
        stopTime = turnTime_start + turnTime

        PV = []
        for i in range(1, 10):
            depth_time_start = time.time()
            bestVal, bestSequence = alphaBeta(board, i, -99999, 99999, board.turn, [], PV)
            PV = bestSequence
            PV_string = ""
            for i in range(len(PV)):
                PV_string += chess.Move.uci(PV[i]) + " "
            print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "pv", PV_string, flush=True)

            if(time.time() > stopTime):
                break
        print("bestmove", bestSequence[0])