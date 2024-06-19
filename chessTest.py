import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import math

def fen_to_bitboards(fen):
    # Definice mapování figur na indexy bitboardů
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    bitboards = [0] * 12

    parts = fen.split()
    board_part = parts[0]

    rank = 7
    file = 0
    for char in board_part:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            index = piece_to_index[char]
            bit_position = rank * 8 + file
            bitboards[index] |= (1 << bit_position)
            file += 1

    input_tensor = np.zeros((12,8,8))
    for i, board in enumerate(bitboards):
        for bit_position in range(64):
            rank = bit_position // 8
            file = bit_position % 8
            if (board & (1 << bit_position)) != 0:
                input_tensor[i, rank, file] = 1
    return input_tensor

def fen_eval_stockfish(fen):
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    return eval.get('value')

def read_fens(input_file, batch_size):
    with open(input_file, 'r') as file:
        fens = [next(file).strip() for _ in range(batch_size)]
    return fens
def process_multiple_fens_to_bit_board(fens):
    bitboards = np.zeros((len(fens), 12, 8, 8), dtype=np.float32)
    for i, fen in enumerate(fens):
        bitboards[i] = fen_to_bitboards(fen)
    return bitboards

def process_fens_into_evals(fens):
    evals_bitboard = np.zeros((len(fens), 1), dtype=np.float32)
    for i, fen in enumerate(fens):
        evals_bitboard[i] = fen_eval_stockfish(fen)
    return evals_bitboard

fens = read_fens(input_file='fens.txt', batch_size=100)
input_bitboards = process_multiple_fens_to_bit_board(fens)
testing_bitboards = process_fens_into_evals(fens)
print(input_bitboards)
print(testing_bitboards)







fen = "rnbqk1nr/pp2p1bp/3p2p1/2pP1p2/2P5/2N2N2/PP2PPPP/R1BQKB1R w KQkq -"


input_tensor = fen_to_bitboards(fen)
print(input_tensor)

stockfish = Stockfish("C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish.set_fen_position(fen)
print(stockfish.get_board_visual())
print(stockfish.get_best_move())
print(stockfish.get_evaluation())
eval = stockfish.get_evaluation()
#y = eval.get("value")
#y = np.array(y)
#y_reshaped = y.reshape(1, 1)
#input_tensor_reshaped = input_tensor.reshape(100, 12, 8, 8)
#print(y)
print(input_tensor.shape)


evalModel = Sequential()
evalModel.add(layers.Conv2D(32, (3,3), input_shape=(12,8,8), activation='relu'))
evalModel.add(layers.Conv2D(32, (3,3), activation='relu'))
evalModel.add(layers.Conv2D(32, (3,3), activation='relu'))
evalModel.add(layers.Flatten())
evalModel.add(layers.Dense(64, activation='relu'))
evalModel.add(layers.Dense(1, activation='relu'))


evalModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
evalModel.fit(input_tensor_reshaped, y_reshaped, batch_size=100, epochs=8)