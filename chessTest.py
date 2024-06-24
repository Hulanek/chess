import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import math
import random

import Functions

fens, evals = Functions.read_database("fen_extractor_output.txt", 50000)
input_bitboards = Functions.process_multiple_fens_to_bit_board(fens)

test_fens, test_evals = Functions.read_database("fen_extractor_output.txt", 50000)
test_input_bitboards = Functions.process_multiple_fens_to_bit_board(test_fens)

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
simple_model_history = evalModel.fit(input_bitboards, evals, batch_size=100, epochs=20, validation_data=(test_input_bitboards, test_evals))

evalModel.save('secondModel.keras')
