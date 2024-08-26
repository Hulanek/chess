import chess
import chess.polyglot
           # q    n    b    r    q    k
mvv_lva = [[105, 205, 305, 405, 505, 605],  #k
            [104, 204, 304, 404, 504, 604], #q
            [103, 203, 303, 403, 503, 603], #r
            [102, 202, 302, 402, 502, 602], #b
            [101, 201, 301, 401, 501, 601], #n
            [100, 200, 300, 400, 500, 600]] #p

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
def move_ordering(legals, board, tt):

    moves_scored = {}

    for move in legals:
        if board.is_capture(move) and not board.is_en_passant(move):
            moves_scored[move] = capture_move_score(board, move)

        else:
            moves_scored[move] = 0

    # try to get best move from tt if its there give it a big value
    zobrist_key = chess.polyglot.zobrist_hash(board)
    TEntry, isSameZobrist = tt.readEntry(zobrist_key)

    # forcing best move from tt to be first (if tt contains info about the move its surely deeper search)
    #if isSameZobrist and TEntry.best_move != None:
    #     if TEntry.best_move in moves_scored:
    #         #print("forcing move", TEntry.best_move, "as a move to try first with depth of", TEntry.depth)
    #         moves_scored[TEntry.best_move] = 99999
    return moves_scored


# Rule that is assuming that is capturing stronger piece with weaker piece is better
def capture_move_score(board, move):
    fromPiece = board.piece_type_at(move.from_square)
    toPiece = board.piece_type_at(move.to_square)
    score = mvv_lva[fromPiece - 1][toPiece - 1]
    return score
