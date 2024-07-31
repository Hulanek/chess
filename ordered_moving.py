import chess
def move_ordering(legals, board):
    good_moves_dict = {}
    chess_piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 11, 'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 11}
    for move in legals:
        if board.is_capture(move) or board.is_en_passant(move):
            fromPiece = board.piece_at(move.from_square)
            toPiece = board.piece_at(move.to_square)
            if chess_piece_values[fromPiece.symbol()] < chess_piece_values[toPiece.symbol()]:
                score = chess_piece_values[toPiece.symbol()] - chess_piece_values[fromPiece.symbol()]
                good_moves_dict[move] = score
            else:
                good_moves_dict[move] = 0
        else:
            good_moves_dict[move] = 0
    return good_moves_dict

board = chess.Board()
legals = board.legal_moves
legal_moves = move_ordering(legals,board)
legal_moves = sorted(legal_moves.items(), key=lambda item: item[1], reverse=True)
ordered_moves = list(*zip(legal_moves)[0])
print(ordered_moves)
for move in legal_moves:
    print(type(move))
    print(move)




