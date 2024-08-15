import chess

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
def move_ordering(legals, board, PV, move_sequence):

    moves_scored = {}

    for move in legals:
        if board.is_capture(move) and not board.is_en_passant(move):
            moves_scored[move] = capture_move_score(board, move)
        elif move.promotion:
            moves_scored[move] = 10000
        else:
            moves_scored[move] = 0
    if PV:
        pv_move = PV[len(move_sequence)] if len(PV) > len(move_sequence) else None
        if pv_move in legals:
            moves_scored.update({pv_move:99999})

    return moves_scored


# Rule that is assuming that is capturing stronger piece with weaker piece is better
def capture_move_score(board, move):
    fromPiece = board.piece_at(move.from_square)
    toPiece = board.piece_at(move.to_square)
    score = piece_value[toPiece.symbol()] - piece_value[fromPiece.symbol()]
    return score

