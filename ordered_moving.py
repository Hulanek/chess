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
