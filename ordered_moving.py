import chess
def move_ordering(legals, board, PV, move_sequence):

    good_moves_dict = {}
    chess_piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 11, 'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 11}
    for move in legals:
        if board.is_capture(move) and not board.is_en_passant(move):
            fromPiece = board.piece_at(move.from_square)
            toPiece = board.piece_at(move.to_square)
            if chess_piece_values[fromPiece.symbol()] < chess_piece_values[toPiece.symbol()]:
                score = chess_piece_values[toPiece.symbol()] - chess_piece_values[fromPiece.symbol()]
                good_moves_dict[move] = score
            else:
                good_moves_dict[move] = 0
        else:
            good_moves_dict[move] = 0

    if PV:
        pv_move = PV[len(move_sequence)] if len(PV) > len(move_sequence) else None
        if pv_move in legals:
            good_moves_dict.update({pv_move:9999})

    return good_moves_dict

