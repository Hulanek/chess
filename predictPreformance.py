import random
import chess
import struct
from bitarray import bitarray
import numpy as np
import Functions
import keras
from chess import pgn
import time
import tensorflow as tf
from onnxruntime import InferenceSession


sess = InferenceSession('new_bitboards_model.onnx')


# totok kdyz zrychlime tak to bude supa
# !! je to zrcadlove ale nemelo by to vadit !!
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

board = chess.Board()
#board.set_fen('5k2/4p2p/4p2p/5pp1/8/1PP5/P2P4/1K6 b - - 0 1')
bitboards = boardToBitboard(board)
bitboards = bitboards.reshape(1, 15, 8, 8)
print(sess.run(None, {'input': bitboards}))
print(bitboards)


start = time.time()
for i in range(100000):
    bitboards = boardToBitboard(board)
    #bitboards = bitboards.reshape(1, 15, 8, 8)
    #sess.run(None, {'input': bitboards})[0]
    #print(bitboards)
end = time.time()
print(end - start, "nps", (100000 / (end - start)), "one node checked in ", (((end - start) / 100000) * 1000000), "microseconds")

start = time.time()
for i in range(100000):
    bitboards = Functions.fen_to_bitboards(board.fen())
end = time.time()
print(end - start)

