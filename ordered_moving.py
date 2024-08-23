import chess
import chess.polyglot

piece_value = {
    'p': 100,
    'r': 500,
    'n': 320,
    'b': 330,
    'q': 900,
    'k': 20000,
    'P': 100,
    'R': 500,
    'N': 320,
    'B': 330,
    'Q': 900,
    'K': 20000,
}
def move_ordering(legals, board, PV, tt):

    moves_scored = {}
    for move in legals:
        board.push(move)
        zobrist_key = chess.polyglot.zobrist_hash(board)
        board.pop()
        TEntry, isSameZobrist = tt.readEntry(zobrist_key)
        if isSameZobrist:
            moves_scored[move] = TEntry.evaluation
            continue # do not remove move from legals
        moves_scored[move] = 900

    # try to get best move from tt if its there give it a big value
    zobrist_key = chess.polyglot.zobrist_hash(board)
    TEntry, isSameZobrist = tt.readEntry(zobrist_key)

    # forcing best move from tt to be first (if tt contains info about the move its surely deeper search)
    if isSameZobrist and TEntry.best_move != None:
        if TEntry.best_move in moves_scored:
            #print("forcing move", TEntry.best_move, "as a move to try first with depth of", TEntry.depth)
            moves_scored[TEntry.best_move] = 99999
    return moves_scored

# PV, TT, KILLER
def order_together(board, killer_moves, depth, original_depth, tt, PV):
    #TRANSPOSITION TABLE HITS FIRST ---- get hits into table/dict and sort them
    legals = list(board.legal_moves)
    ordered_moves = move_ordering(legals, board, PV, tt)
    # sorted_tt_hits_evaluated = sorted(unsorted_tt_hits.items(), key=lambda item: item[1], reverse=True)


   # CAPTURES SECOND    ----- sort captures and update the dict
    for move in legals:
        if board.is_legal(move) and not board.is_en_passant(move):
            if move not in ordered_moves.keys():
                ordered_moves[move] = capture_move_score(board, move)

    # delete captures and tt from legals
    legals = [move for move in legals if move not in ordered_moves.keys()]


    #TWO KILLER MOVES ------ update and all others should be below these 3
    for killer in killer_moves[original_depth - depth]:
        if killer in legals:
            i = 1
            if killer in legals and board.is_legal(killer):
                if killer not in ordered_moves.keys():
                    ordered_moves[killer] = i * 100
                    legals.remove(killer)
                    i += 1

    return ordered_moves



# Rule that is assuming that is capturing stronger piece with weaker piece is better
def capture_move_score(board, move):
    fromPiece = board.piece_at(move.from_square)
    toPiece = board.piece_at(move.to_square)
    score = piece_value[toPiece.symbol()] - piece_value[fromPiece.symbol()]
    return score if score > 0 else 0
