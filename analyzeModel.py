import chess
import keras
from keras import Sequential
from keras import layers
import numpy as np
from stockfish import Stockfish
import math
import Functions
#from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
import random


def averageMiss(model, databaseFile, numOfSamples):
    test_fens, test_evals = Functions.read_database(databaseFile, numOfSamples)  # offset has to be even number
    test_bitboards = Functions.process_multiple_fens_to_bit_board(test_fens)
    predictions = model.predict(test_bitboards)
    modelEvalMiss = 0
    guessEvalMiss = 0
    for i in range(numOfSamples):
        modelEvalMiss += abs(test_evals[i] - predictions[i])
        guessEvalMiss += abs(test_evals[i] - 0)

    modelEvalAvgMiss = (modelEvalMiss / numOfSamples) * 1000
    guessEvalAvgMiss = (guessEvalMiss / numOfSamples) * 1000
    print("Average model miss - ", modelEvalAvgMiss, "Average random guess miss - ", guessEvalAvgMiss)


model1 = keras.saving.load_model('firstModel.keras')

averageMiss(model1, "fens_evals_first_half.txt", 10000)
averageMiss(model1, "fens_evals_second_half.txt", 10000)


model2 = keras.saving.load_model('ULTIAMTEMODEL3000.keras')

averageMiss(model2, "fens_evals_first_half.txt", 10000)
averageMiss(model2, "fens_evals_second_half.txt", 10000)