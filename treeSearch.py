import random
import chess
import ordered_moving
from onnxruntime import InferenceSession
import Functions
import time
from chess import polyglot
from transpositionTable import TT

test_fens = ["1k6/8/5ppp/8/5PPP/8/8/1K6 w HAha - 0 1",   # 3v3 pawns
            "1k6/8/8/8/8/8/6Q1/3K4 w - - 0 1", # queen mat
            "8/p4p2/1ppk2p1/4p2p/8/2P1PPP1/P1P4P/3K4 w - - 0 1", # pawns fight
            "8/8/3k4/8/8/1P6/1K6/8 w - - 1 2", #pawn promotion
            "r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1 w q - 0 1",
            "r1n2N1k/2n2K1p/3pp3/5Pp1/b5R1/8/1PPP4/8 w - - 0 1",
            "r1b1r1k1/1pqn1pbp/p2pp1p1/P7/1n1NPP1Q/2NBBR2/1PP3PP/R6K w - -",
            "5b2/p2k1p2/P3pP1p/n2pP1p1/1p1P2P1/1P1KBN2/7P/8 w - -",
            "r3kbnr/1b3ppp/pqn5/1pp1P3/3p4/1BN2N2/PP2QPPP/R1BR2K1 w kq - 0 1",
            "r2r2k1/1p1n1pp1/4pnp1/8/PpBRqP2/1Q2B1P1/1P5P/R5K1 b - -"]

sess = InferenceSession('2conv32_33_3dense_768_256.onnx')
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
    global num_of_nodes
    num_of_nodes = num_of_nodes + 1
    bitboards = Functions.boardToBitboard(board)
    bitboards = bitboards.reshape(1, 13, 8, 8)
    return sess.run(None, {'input': bitboards})[0][0][0]

def store_killer_move(current_move, depth, killer_moves):
    if current_move not in killer_moves[depth]:
        if len(killer_moves[depth]) == 2:
            killer_moves[depth][1] = killer_moves[depth][0]

        killer_moves[depth][0] = current_move

killer_moves = [[] for _ in range(10)]

def alphaBeta(board, depth, original_depth, alpha, beta, move_sequence, PV, stopTime):
    alphaOrig = alpha
    zobrist_key = chess.polyglot.zobrist_hash(board)
    ttEntry, isSameZobrist = tt.readEntry(zobrist_key)

    # tt cutoff
    if isSameZobrist and ttEntry.depth >= depth and ttEntry.best_move != None:
        if ttEntry.flag == EXACT:
            return ttEntry.evaluation, move_sequence + [ttEntry.best_move]
        elif ttEntry.flag == LOWERBOUND:
            alpha = max(alpha, ttEntry.evaluation)
        elif ttEntry.flag == UPPERBOUND:
            beta = min(beta, ttEntry.evaluation)
        if alpha >= beta:
            return ttEntry.evaluation, move_sequence + [ttEntry.best_move]


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
            TEntry, isSameZobrist = tt.readEntry(zobrist_key)
            if isSameZobrist:
                num_of_nodes += 1
                return TEntry.evaluation, move_sequence

            evaluation = eval(board)
            tt.writeEntry(zobrist_key, evaluation, None, depth, 0, board.fullmove_number)
            return evaluation, move_sequence
        else:
            TEntry, isSameZobrist = tt.readEntry(zobrist_key)
            if isSameZobrist:
                num_of_nodes += 1
                return -TEntry.evaluation, move_sequence

            evaluation = eval(board)
            tt.writeEntry(zobrist_key, evaluation, None, depth, 0, board.fullmove_number)
            return -evaluation, move_sequence

    bestVal = -99999
    bestSequence = move_sequence

    # generating moves
    legals = list(board.legal_moves)

    # scoring moves
    ordered_moves = ordered_moving.move_ordering(legals, board, PV, move_sequence, tt)

    #Ordered moves conclusion

    # sorting moves by score
    # better would be to take the best value in each iteration (you dont have to sort moves that you wont use)
    ordered_moves = sorted(ordered_moves.items(), key=lambda item: item[1], reverse=True)
    #print(depth, ordered_moves)


    for move in ordered_moves:
        #print(move, depth)
        #time check
        if (num_of_nodes % TIME_NUM_OF_NODES == 0):
            if(time.time() > stopTime):
                break

        board.push(move[0])
        newEval, newSequence = alphaBeta(board, depth - 1, original_depth, -beta, -alpha, move_sequence + [move[0]], PV, stopTime)
        newEval = -newEval
        board.pop()

        if newEval > bestVal:
            bestVal = newEval
            bestSequence = newSequence
        alpha = max(alpha, newEval)
        if alpha >= beta:
            if len(killer_moves[original_depth - depth]) < 2:
                # Update if less than 2
                killer_moves[original_depth - depth].append(move)
            else:
                # if 2(full) -> update
                store_killer_move(move, original_depth - depth, killer_moves)
            break

    # tt write
    ttFlag = 0
    if bestVal <= alphaOrig:
        ttFlag = UPPERBOUND
    elif bestVal >= beta:
        ttFlag = LOWERBOUND
    else:
        ttFlag = EXACT

    tt.writeEntry(zobrist_key, bestVal, bestSequence[original_depth - depth], depth, ttFlag, board.fullmove_number)
    return bestVal, bestSequence

for fen_index in range(0, len(test_fens)):
    print("---")
    tt = TT(65536)
    num_of_nodes = 0
    board = chess.Board(test_fens[fen_index])
    PV = []
    stopTime = time.time() + 10
    turnTime_start = time.time()
    for i in range(1, 8):
        if (time.time() > stopTime):
            break
        bestVal, bestSequence = alphaBeta(board, i, i, -99999, 99999, [], PV, stopTime)
        PV = bestSequence
        PV_string = ""
        for j in range(len(PV)):
            PV_string += chess.Move.uci(PV[j]) + " "
        print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "nps", num_of_nodes/(time.time() - turnTime_start), "pv", PV_string)
        #tt.contentOfTT()
        #tt.infoTT()
    end = time.time()
    print("time taken", end - turnTime_start)
    print("bestmove", PV[0])




# UCI

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
        num_of_nodes = 0
        PV = []
        for i in range(1, 10):
            #time check
            if (time.time() > stopTime):
                break
            else:
                bestVal, bestSequence = alphaBeta(board, i, i, -99999, 99999, [], PV, stopTime)
                PV = bestSequence
                PV_string = ""
                for j in range(len(PV)):
                    PV_string += chess.Move.uci(PV[j]) + " "
            print("info depth", i, "score cp", int(bestVal * 1000), "nodes", num_of_nodes, "nps", num_of_nodes/(time.time() - turnTime_start + 1),"pv", PV_string, flush=True)

            #print("depth time taken", time.time() - turnTime_start)
        print("bestmove", bestSequence[0])