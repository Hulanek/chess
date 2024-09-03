import random
import chess
import ordered_moving
import Functions
from stockfish import Stockfish
import time
from chess import polyglot
from transpositionTable import TT
import evaluation


test_fens = ["r1bqkb1r/ppp1pppp/2n5/3n4/2QP4/4PN2/PP3PPP/RNB1KB1R b KQkq - 0 6",
            "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/1PP1NPB1/R1BQ1RK1 w - - 0 9",#video fen
            #"1k6/8/5ppp/8/5PPP/8/8/1K6 w HAha - 0 1",   # 3v3 pawns
            #"1k6/8/8/8/8/8/6Q1/3K4 w - - 0 1", # queen mat
            #"8/p4p2/1ppk2p1/4p2p/8/2P1PPP1/P1P4P/3K4 w - - 0 1", # pawns fight
            #"8/8/3k4/8/8/1P6/1K6/8 w - - 1 2", #pawn promotion
            "r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1 w q - 0 1",
            "r1n2N1k/2n2K1p/3pp3/5Pp1/b5R1/8/1PPP4/8 w - - 0 1",
            "r1b1r1k1/1pqn1pbp/p2pp1p1/P7/1n1NPP1Q/2NBBR2/1PP3PP/R6K w - -",
            "5b2/p2k1p2/P3pP1p/n2pP1p1/1p1P2P1/1P1KBN2/7P/8 w - -",
            "r3kbnr/1b3ppp/pqn5/1pp1P3/3p4/1BN2N2/PP2QPPP/R1BR2K1 w kq - 0 1",
            "r2r2k1/1p1n1pp1/4pnp1/8/PpBRqP2/1Q2B1P1/1P5P/R5K1 b - -"]

#sess = InferenceSession('2conv32_33_3dense_768_256.onnx')
TIME_NUM_OF_NODES = 1024
UPPERBOUND = 1
LOWERBOUND = 2
EXACT = 3
num_of_nodes = 0
sum_of_nodes = 0
tt = TT(2 ** 16) # 2^16 = 65536


def randomPlay(board):
    legals = list(board.legal_moves)
    return legals[random.randint(0, len(legals)-1)]

def eval(board):
    bitboards = Functions.boardToBitboard(board)
    bitboards = bitboards.reshape(1, 13, 8, 8)
    #return fen_eval_stockfish(board.fen())
    #return sess.run(None, {'input': bitboards})[0][0][0]
    return evaluation.staticEvaluation(board)

def quiescenceSearch(board, alpha, beta, move_sequence):
    global num_of_nodes
    num_of_nodes = num_of_nodes + 1
    if board.turn == chess.WHITE:
        evaluation = eval(board)
    else:
        evaluation = -eval(board)
    if evaluation >= beta:
        return beta, move_sequence
    if evaluation > alpha:
        alpha = evaluation
    captureLegals = list(board.legal_moves)

    i = 0
    while i < len(captureLegals):
        if not board.is_capture(captureLegals[i]) and not board.is_en_passant(captureLegals[i]): # en passant chybi
            captureLegals.remove(captureLegals[i])
        else:
            i += 1

    if(len(captureLegals) == 0):
        return evaluation, move_sequence

    scoredCaptures = ordered_moving.move_ordering(captureLegals, board, tt, PV)
    captureLegals = sorted(scoredCaptures.items(), key=lambda item: item[1], reverse=True)
    newSequence = move_sequence

    for move in captureLegals:

        # time control
        if (num_of_nodes % TIME_NUM_OF_NODES == 0):
           if (time.time() > stopTime):
               return alpha, move_sequence

        board.push(move[0])
        newEval, newSequence = quiescenceSearch(board, -beta, -alpha, move_sequence + [move[0]])
        newEval = -newEval
        board.pop()

        if (newEval >= beta):
            return beta, newSequence
        if (newEval > alpha):
            alpha = newEval


    return alpha, newSequence



def alphaBeta(board, depth, original_depth, alpha, beta, move_sequence, PV, stopTime, nullMove):
    alphaOrig = alpha
    zobrist_key = chess.polyglot.zobrist_hash(board)
    # looking to tt if there is current node
    ttEntry, isSameZobrist = tt.readEntry(zobrist_key)

    # three-fold repetition condition
    if board.is_repetition():
        return 0, move_sequence

    # tt cutoff
    if isSameZobrist and ttEntry.depth >= depth and ttEntry.best_move != None:
        if ttEntry.flag == EXACT:
            pass
            #return ttEntry.evaluation, move_sequence + [ttEntry.best_move]
        elif ttEntry.flag == LOWERBOUND:
            alpha = max(alpha, ttEntry.evaluation)
        elif ttEntry.flag == UPPERBOUND:
            beta = min(beta, ttEntry.evaluation)
        if alpha >= beta:
            pass
            #return ttEntry.evaluation, move_sequence + [ttEntry.best_move]

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -999999, move_sequence
        else:
            return -999999, move_sequence
    if board.is_stalemate():
        return 0, move_sequence

    # for using this condition (breaking repetition) you need at least depth 2
    if board.is_repetition():
        return 0, move_sequence
    if depth == 0:
        # eval from quiet search is necessary
        quietEval, move_sequence = quiescenceSearch(board, alpha, beta, move_sequence)
        return quietEval, move_sequence

    # null move pruning (broken)
    # if board.is_check() == False and nullMove:
    #     staticEval = eval(board)
    #
    #     if depth > 2 and staticEval >= beta:
    #         board.push(chess.Move.from_uci('0000'))
    #         newEval, newSequence = alphaBeta(board, depth - 2 - 1, original_depth,-beta, -beta + 1, move_sequence, PV, stopTime, False)
    #         newEval = -newEval
    #         board.pop()
    #         if newEval >= beta:
    #             return beta, newSequence


    bestVal = -1000000
    bestSequence = move_sequence

    # generating moves
    legals = list(board.legal_moves)

    # scoring moves
    ordered_moves = ordered_moving.move_ordering(legals, board, tt, PV)

    # sorting moves by score
    # better would be to take the best value in each iteration (you dont have to sort moves that you wont use)
    ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)

    for move in ordered_moves:
        # time control
        if (num_of_nodes % TIME_NUM_OF_NODES == 0):
            if(time.time() > stopTime):
                break

        board.push(move[0])
        newEval, newSequence = alphaBeta(board, depth - 1, original_depth, -beta, -alpha, move_sequence + [move[0]], PV, stopTime, True)
        newEval = -newEval
        board.pop()

        if newEval > bestVal:
            bestVal = newEval
            bestSequence = newSequence
        alpha = max(alpha, newEval)
        if alpha >= beta:
            break

    # setting flag of tt entry
    ttFlag = 0
    if bestVal <= alphaOrig:
        ttFlag = UPPERBOUND
    elif bestVal >= beta:
        ttFlag = LOWERBOUND
    else:
        ttFlag = EXACT
    # writing result to tt
    # condition if time is up
    if (len(bestSequence) - 1 >= original_depth - depth):
        tt.writeEntry(zobrist_key, bestVal, bestSequence[original_depth - depth], depth, ttFlag, board.fullmove_number)
    return bestVal, bestSequence




'''

for fen_index in range(0, len(test_fens)):
    print("---")
    tt = TT(2 ** 16)
    num_of_nodes = 0
    board = chess.Board(test_fens[fen_index])
    PV = []
    stopTime = time.time() + 300000
    turnTime_start = time.time()
    for i in range(1, 15):
        if (time.time() > stopTime):
            break
        bestVal, bestSequence = alphaBeta(board, i, i, -99999, 99999, [], PV, stopTime, True)
        PV = bestSequence
        PV_string = ""
        for j in range(len(PV)):
            PV_string += chess.Move.uci(PV[j]) + " "
        print("info depth", i, "score cp", int(bestVal), "nodes", num_of_nodes, "nps", int(num_of_nodes/(time.time() - turnTime_start + 0.001)), "pv", PV_string)
        #tt.contentOfTT()
        #tt.infoTT()
    end = time.time()
    print("time taken", end - turnTime_start)
    print("bestmove", PV[0])
'''









# UCI

board = chess.Board()

while True:
    args = input().split()

    if args[0] == "uci":
        #print("model_name")
        print("id name Supr Engine 3000")
        print("id author Martinek a Honzik")
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

        #tt = TT(2 ** 16)
        # chybi inkrementovaci cas

        if len(args) == 7:
            wtime, btime, moves= [int(a) / 1000 for a in args[2::2]]
        elif len(args) == 3:
            wtime = int(args[2]) / 1000 * 30
            btime = int(args[2]) / 1000 * 30
        elif len(args) == 9:
            wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
        elif len(args) == 11:
            wtime, btime, winc, binc, moves= [int(a) / 1000 for a in args[2::2]]

        if(board.turn == chess.WHITE):
            timeLeft = wtime
        else:
            timeLeft = btime

        turnTime = timeLeft / 30  # allocate 1/30th part of time left for current turn
        #print(turnTime)
        turnTime_start = time.time()
        stopTime = turnTime_start + turnTime
        #print(time.time(), stopTime)
        num_of_nodes = 0
        PV = []
        for i in range(1, 20):
            #time check


            bestVal, bestSequence = alphaBeta(board, i, i, -99999, 99999, [], PV, stopTime, True)
            if (time.time() > stopTime):
                break
            PV = bestSequence
            PV_string = ""
            for j in range(len(PV)):
                PV_string += chess.Move.uci(PV[j]) + " "
            print("info depth", i, "score cp", int(bestVal), "nodes", num_of_nodes, "nps", int(num_of_nodes/(time.time() - turnTime_start + 1)),"pv", PV_string, flush=True)

            #print("depth time taken", time.time() - turnTime_start)
        print("bestmove", PV[0])