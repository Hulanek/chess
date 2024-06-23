import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import math
#from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
import random

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

stockfish = Stockfish("C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe")

def fen_eval_stockfish(fen):
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    return eval.get('value')/1000

def read_database(input_file, batch_size, offset):
    fens = []
    evals = np.zeros((batch_size,1))
    numOfLines = 0
    with open(input_file, 'r') as file:

        #offset for test data
        linesToSkip = 0
        while linesToSkip < offset:
            line = file.readline()
            linesToSkip += 1
        #-----

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


def process_multiple_fens_to_bit_board(fens):
    bitboards = np.zeros((len(fens), 15, 8, 8), dtype=np.float32)
    for i, fen in enumerate(fens):
        bitboards[i] = fen_to_bitboards(fen)
    return bitboards

def process_fens_into_evals(fens):
    evals_bitboard = np.zeros((len(fens), 1), dtype=np.float32)
    for i, fen in enumerate(fens):
        evals_bitboard[i] = fen_eval_stockfish(fen)
    return evals_bitboard


fens, evals = read_database("fen_extractor_output.txt", 50000, 0)
input_bitboards = process_multiple_fens_to_bit_board(fens)

test_fens, test_evals = read_database("fen_extractor_output.txt", 50000, 100000)# offset has to be even number
test_input_bitboards = process_multiple_fens_to_bit_board(test_fens)


evalModel = Sequential()
evalModel.add(layers.Conv2D(32, (3, 3), input_shape=(15, 8, 8), activation='relu'))
evalModel.add(layers.Conv2D(32, (3, 3), activation='relu'))
#evalModel.add(layers.Dropout(0.25))_
#evalModel.add(layers.Conv2D(32, (3, 3), activation='relu'))
#evalModel.add(layers.Dropout(0.25))
evalModel.add(layers.Flatten())
evalModel.add(layers.Dense(768, activation='relu'))
evalModel.add(layers.Dense(128, activation='relu'))
evalModel.add(layers.Dropout(0.5))
evalModel.add(layers.Dense(1, activation='tanh'))

evalModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
simple_model_history = evalModel.fit(input_bitboards, evals, batch_size=100, epochs=20, validation_data=(test_input_bitboards, test_evals))

#evalModel.save('firstModel.keras')

predictions = evalModel.predict(test_input_bitboards)
approx = 0
approx_random = 0
averageEval = 0
for i in range(1000):
    approx += abs(test_evals[i] - predictions[i])
    averageEval += test_evals[i]
    approx_random += abs(test_evals[i] - 0)
    print(f'Expected value:{test_evals[i]}, predicted values:  {predictions[i]}')

# approx = approx / 1000 (divide by number of examples) * 1000 (multiply to compensate previous divide)
averageEval = averageEval / 1000
print(approx)
approx_random = approx_random
print(approx_random)
print(averageEval)
