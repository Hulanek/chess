import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import math
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
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

stockfish = Stockfish("C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe")

def fen_eval_stockfish(fen):
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    return eval.get('value')/1000

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

nn_test_fens = read_fens(input_file='fens_test.txt', batch_size=100)
nn_test_fens_bitboard = process_multiple_fens_to_bit_board(nn_test_fens)
test_fens_eval_bitboards = process_fens_into_evals(nn_test_fens)


#input_bitboards_train, input_bitboards_val, testing_bitboards_train, testing_bitboards_val = train_test_split(input_bitboards, testing_bitboards, test_size=0.2, random_state=42)


evalModel = Sequential()
evalModel.add(layers.Conv2D(32, (3, 3), input_shape=(12, 8, 8), activation='relu'))
evalModel.add(layers.Conv2D(32, (3, 3), activation='relu'))
#evalModel.add(layers.Dropout(0.25))
#evalModel.add(layers.Conv2D(32, (3, 3), activation='relu'))
#evalModel.add(layers.Dropout(0.25))
evalModel.add(layers.Flatten())
evalModel.add(layers.Dense(768, activation='relu'))
evalModel.add(layers.Dense(128, activation='relu'))
evalModel.add(layers.Dropout(0.5))
evalModel.add(layers.Dense(1, activation='tanh'))

evalModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
simple_model_history = evalModel.fit(input_bitboards, testing_bitboards, batch_size=10, epochs=20, validation_data=(input_bitboards, testing_bitboards))

predictions = evalModel.predict(nn_test_fens_bitboard)
approx = 0
approx_random = 0
for i in range(100):
    approx += abs(test_fens_eval_bitboards[i] - predictions[i])
    approx_random += abs(test_fens_eval_bitboards[i] - random.randint(-1000,1000)/1000)
    print(f'Expected value:{test_fens_eval_bitboards[i]}, predicted values:  {predictions[i]}')

approx = approx/100
print(approx)
approx_random = approx_random/100
print(approx_random)

approx = approx*1000
print(approx)
approx_random = approx_random*1000
print(approx_random)

