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
def move_ordering(legals, board, PV, move_sequence, tt):

    moves_scored = {}
    for move in legals:
        board.push(move)
        zobrist_key = chess.polyglot.zobrist_hash(board)
        board.pop()
        TEntry, isSameZobrist = tt.readEntry(zobrist_key)
        if isSameZobrist:
            moves_scored[move] = TEntry.evaluation
            continue # do not remove move from legals
        moves_scored[move] = 0

    # try to get best move from tt if its there give it a big value
    zobrist_key = chess.polyglot.zobrist_hash(board)
    TEntry, isSameZobrist = tt.readEntry(zobrist_key)

    # forcing best move from tt to be first (if tt contains info about the move its surely deeper search)
    if isSameZobrist and TEntry.best_move != None:
        if TEntry.best_move in moves_scored:
            #print("forcing move", TEntry.best_move, "as a move to try first with depth of", TEntry.depth)
            moves_scored[TEntry.best_move] = 99999
    return moves_scored


def order_together(board, killer_moves, depth, original_depth, tt_move=None):
    #TRANSPOSITION TABLE HITS FIRST ---- get hits into table/dict and sort them
    ordered_moves = []
    legals = list(board.legal_moves)


    if tt_move in legals:
        if board.is_legal(tt_move):
            ordered_moves = [tt_move]  # and value ?
            legals.remove(tt_move)

    #CAPTURES SECOND    ----- sort captures and update the dict
    captures = [(move, capture_move_score(board, move)) for move in legals if board.is_capture(move) and not board.is_en_passant(move)]
    captures.sort(key= lambda pair: pair[1], reverse=True)

    captures_moves = [move for move, _ in captures if move != tt_move]
    legals = [move for move in legals if move not in captures_moves]
    ordered_moves.extend(captures_moves)

    #TWO KILLER MOVES ------ update and all others should be below these 3
    for killer in killer_moves[original_depth - depth]:
        if killer in legals:
            if killer in legals and board.is_legal(killer):
                if killer not in ordered_moves or killer != tt_move:
                    ordered_moves.append(killer)
                    legals.remove(killer)

    ordered_moves.extend(legals)

    return ordered_moves



# Rule that is assuming that is capturing stronger piece with weaker piece is better
def capture_move_score(board, move):
    fromPiece = board.piece_at(move.from_square)
    toPiece = board.piece_at(move.to_square)
    score = piece_value[toPiece.symbol()] - piece_value[fromPiece.symbol()]
    return score if score > 0 else 0
