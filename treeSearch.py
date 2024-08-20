import random
import chess
import ordered_moving
from onnxruntime import InferenceSession
import Functions
import time
from bit_tryer import Transposition_table2
from chess import polyglot
from transpositionTable import TT

import threading

sess = InferenceSession('2conv32_33_3dense_768_256.onnx')
TIME_NUM_OF_NODES = 1024
num_of_nodes = 0
sum_of_nodes = 0
tt = TT(128)

def randomPlay(board):
    legals = list(board.legal_moves)
    return legals[random.randint(0, len(legals)-1)]

def eval(board):
    global num_of_nodes
    num_of_nodes = num_of_nodes + 1
    bitboards = Functions.boardToBitboard(board)
    bitboards = bitboards.reshape(1, 13, 8, 8)
    return sess.run(None, {'input': bitboards})[0][0][0]

def trans_table_add2(zobrist, eval, depth, trans_table, move_sequence, entry_type):
    trans_table.set(zobrist, eval, depth, move_sequence, entry_type)


def alphaBeta(board, depth, original_depth, alpha, beta, move_sequence, PV, stopTime):
    zobrist_key = chess.polyglot.zobrist_hash(board)

    alpha_orig = alpha

    ttEntry, valid = tt.readEntry(zobrist_key)
    if valid:
        if ttEntry.flag == "EXACT" and ttEntry.depth >= original_depth:
            if ttEntry.best_move:
                return ttEntry.evaluation, move_sequence
        #     return ttEntry.evaluation, ttEntry.bestMove
        elif ttEntry.flag == "LOWERBOUND":
            alpha = max(alpha, ttEntry.evaluation)

        elif ttEntry.flag == "UPPERBOUND":
            beta = min(beta, ttEntry.evaluation)
        if alpha >= beta:
            if ttEntry.best_move:
                return ttEntry.evaluation, move_sequence

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -10000, move_sequence
        else:
            return 10000, move_sequence
    if board.is_stalemate():
        return 0, move_sequence
    if depth == 0:
        global num_of_nodes
        if board.turn == chess.WHITE:
            zobrist_key = chess.polyglot.zobrist_hash(board)
            TEntry, isSameZobrist = tt.readEntry(zobrist_key)
            if isSameZobrist:
                num_of_nodes += 1
                return TEntry.evaluation, move_sequence

            evaluation = eval(board)
            tt.writeEntry(zobrist_key, evaluation, move_sequence[original_depth - depth - 1], original_depth, 'EXACT', board.fullmove_number)
            return evaluation, move_sequence
        else:
            zobrist_key = chess.polyglot.zobrist_hash(board)
            TEntry, isSameZobrist = tt.readEntry(zobrist_key)
            if isSameZobrist:
                num_of_nodes += 1
                return -TEntry.evaluation, move_sequence

            evaluation = eval(board)
            tt.writeEntry(zobrist_key, evaluation, move_sequence[original_depth - depth - 1], original_depth, 'EXACT', board.fullmove_number)
            return -evaluation, move_sequence

    bestVal = -99999
    bestSequence = move_sequence

    # generating moves
    legals = list(board.legal_moves)
    # scoring moves
    ordered_moves = ordered_moving.move_ordering(legals, board, PV, move_sequence)

    # sorting moves by score
    # better would be to take the best value in each iteration (you dont have to sort moves that you wont use)
    ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)

    for move in ordered_moves:

        #time check
        if (num_of_nodes % TIME_NUM_OF_NODES == 0):
            if(time.time() > stopTime):
                break

        board.push(move[0])
        newEval, newSequence = alphaBeta(board, depth - 1, original_depth, -beta, -alpha, move_sequence + [move[0]], PV, stopTime)
        newEval = -newEval
        board.pop()
        # if newEval >= beta:
        #     return newEval, bestSequence
        if newEval > bestVal:
            bestVal = newEval
            bestSequence = newSequence
        alpha = max(alpha, newEval)
        if alpha >= beta:
            break

    tt_value = newEval
    if tt_value <= alpha_orig:
        tt_entry_flag = "UPPERBOUND"
    elif tt_value >= beta:
        tt_entry_flag = "LOWERBOUND"
    else:
        tt_entry_flag = "EXACT"

    tt_entry_move = bestSequence[original_depth- depth - 1]
    tt.writeEntry(zobrist_key, tt_value, tt_entry_move, original_depth, tt_entry_flag, board.fullmove_number)

    return bestVal, bestSequence


board = chess.Board('rnb1k1nr/pppp2pp/5q2/4pp2/1b1PP3/2P5/PP1B1PPP/RN1QKBNR w KQkq - 1 5')
PV = []
stopTime = time.time() + 30000
turnTime_start = time.time()
for i in range(1, 9):
    if (time.time() > stopTime):
        break
    bestVal, bestSequence = alphaBeta(board, i, i, -99999, 99999, [], PV, stopTime)
    PV = bestSequence
    PV_string = ""
    for j in range(len(PV)):
        PV_string += chess.Move.uci(PV[j]) + " "
    print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "nps", num_of_nodes/(time.time() - turnTime_start), "pv", PV_string)
    #tt.contentOfTT()
end = time.time()
print("time taken", end - turnTime_start)
print("bestmove", PV[0])



board = chess.Board()

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
            #time check
            if (time.time() > stopTime):
                break
            else:
                bestVal, bestSequence = alphaBeta(board, i, -99999, 99999, [], PV, stopTime)
                PV = bestSequence
                PV_string = ""
                for j in range(len(PV)):
                    PV_string += chess.Move.uci(PV[j]) + " "
            print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "nps", num_of_nodes/(time.time() - turnTime_start),"pv", PV_string, flush=True)

            print("depth time taken", time.time() - turnTime_start)
        print("bestmove", bestSequence[0])