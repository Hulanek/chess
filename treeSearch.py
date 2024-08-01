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
    return sess.run(None, {'input': bitboards})[0]

def alphaBeta(board, depth, alpha, beta, maximize, move_sequence, PV):
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

    # podle tech oficialnich pravidel by to melo jit nastavit podle fenu ale to zatim nemame
    elif args[:2] == ["position", "startpos"]:
        board = chess.Board()
        for ply, move in enumerate(args[3:]): # od indexu 4 - vynecha slovo moves
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
        print(turnTime_start, turnTime)

        PV = []
        for i in range(1, 10):
            depth_time_start = time.time()
            # asi ukladat posledni sekvenci -> ta se asi vyzkousi jako prvni pri dalsi iteraci
            bestVal, bestSequence = alphaBeta(board, i, -99999, 99999, board.turn, [], PV)
            PV = bestSequence
            print("info depth", i, "score cp", int(bestVal[0][0] * 1000), "checkedNodes(not in uci format)", num_of_nodes)

            if(time.time() > stopTime):
                break

        print("bestmove", bestSequence[0])
