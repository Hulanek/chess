import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import time
import math
#from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
import random

stockfish = Stockfish("C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe")

def float_to_binary_array(num):
    # Use bit shifting and masking to create the bit array
    bit_array = np.zeros(64)
    bit_length = num.mask.bit_length()
    for i in range(63, 63 - bit_length, -1):
        bit_array[i] = num.mask & 1
        num.mask >>= 1
    bit_array = bit_array.reshape(8, 8)
    return bit_array

def boardToBitboard(board):
    bitboards = np.zeros((15, 8, 8)).astype(np.float32)

    # order of figures
    #  White - PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    bitboards[0] = float_to_binary_array(board.pieces(chess.PAWN, chess.WHITE))
    bitboards[1] = float_to_binary_array(board.pieces(chess.ROOK, chess.WHITE))
    bitboards[2] = float_to_binary_array(board.pieces(chess.KNIGHT, chess.WHITE))
    bitboards[3] = float_to_binary_array(board.pieces(chess.BISHOP, chess.WHITE))
    bitboards[4] = float_to_binary_array(board.pieces(chess.QUEEN, chess.WHITE))
    bitboards[5] = float_to_binary_array(board.pieces(chess.KING, chess.WHITE))

    #  Black - PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    bitboards[6] = float_to_binary_array(board.pieces(chess.PAWN, chess.BLACK))
    bitboards[7] = float_to_binary_array(board.pieces(chess.ROOK, chess.BLACK))
    bitboards[8] = float_to_binary_array(board.pieces(chess.KNIGHT, chess.BLACK))
    bitboards[9] = float_to_binary_array(board.pieces(chess.BISHOP, chess.BLACK))
    bitboards[10] = float_to_binary_array(board.pieces(chess.QUEEN, chess.BLACK))
    bitboards[11] = float_to_binary_array(board.pieces(chess.KING, chess.BLACK))

    # bitboard of index 12 and 13 is left for enpassant moves

    # castling rights and who is on turn are just on first five bits of 15th bitboard
    bitboards[14][0] = board.has_kingside_castling_rights(chess.WHITE)
    bitboards[14][1] = board.has_queenside_castling_rights(chess.WHITE)
    bitboards[14][2] = board.has_kingside_castling_rights(chess.BLACK)
    bitboards[14][3] = board.has_queenside_castling_rights(chess.BLACK)
    bitboards[14][4] = board.turn
    return bitboards
def fen_to_bitboards(fen):
    # Definice mapování figur na indexy bitboardů
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    bitboards = [0] * 12

    parts = fen.split()
    board_part = parts[0]

    rank = 0
    file = 0
    for char in board_part:
        if char == '/':
            rank += 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            index = piece_to_index[char]
            bit_position = rank * 8 + file
            bitboards[index] |= (1 << bit_position)
            file += 1
    input_tensor = np.zeros((15,8,8))


    for i, board in enumerate(bitboards):
        for bit_position in range(64):
            rank = bit_position // 8
            file = bit_position % 8
            if (board & (1 << bit_position)) != 0:
                input_tensor[i, rank, file] = 1


    board = chess.Board(fen)
    legal_moves = board.legal_moves
    #dictCols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h':7}
    for move in legal_moves:
        tile_index = move.to_square
        col = tile_index % 8
        row = 7 - tile_index // 8
        input_tensor[12, row, col] = 1

    board.turn = not board.turn
    legal_moves = board.legal_moves
    for move in legal_moves:
        tile_index = move.to_square
        col = tile_index % 8
        row = 7 - tile_index // 8
        input_tensor[13, row, col] = 1


    board.turn = not board.turn
    castlingArray = list()
    castlingArray.append(board.has_kingside_castling_rights(chess.WHITE))
    castlingArray.append(board.has_queenside_castling_rights(chess.WHITE))
    castlingArray.append(board.has_kingside_castling_rights(chess.BLACK))
    castlingArray.append(board.has_queenside_castling_rights(chess.BLACK))
    castlingArray.append(board.turn)
    castlingArray.append(board.turn)
    castlingArray.append(board.turn)
    castlingArray.append(board.turn)
    for i in range(8):
        input_tensor[14, i] = castlingArray
    return input_tensor
def read_database(input_file, batch_size):
    fens = []
    evals = np.zeros((batch_size,1))
    numOfLines = 0
    with open(input_file, 'r') as file:
        while numOfLines < batch_size * 2:
            line = file.readline()
            numOfLines += 1
            if(numOfLines % 2 == 0):
                currEval = float(line)
                index = numOfLines // 2
                evals[index - 1] = currEval
            else:
                fens.append(line)
    return fens, evals

def fen_eval_stockfish(fen):
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    return eval.get('value')/1000

def read_fens(input_file, batch_size):
    with open(input_file, 'r') as file:
        fens = [next(file).strip() for _ in range(batch_size)]
    return fens
def process_multiple_fens_to_bit_board(fens):
    bitboards = np.zeros((len(fens), 15, 8, 8), dtype=np.float32)
    for i, fen in enumerate(fens):
        tmp_board = chess.Board()
        tmp_board.set_fen(fen)
        bitboards[i] = boardToBitboard(tmp_board)
    return bitboards

def process_fens_into_evals(fens):
    evals_bitboard = np.zeros((len(fens), 1), dtype=np.float32)
    for i, fen in enumerate(fens):
        evals_bitboard[i] = fen_eval_stockfish(fen)
    return evals_bitboard