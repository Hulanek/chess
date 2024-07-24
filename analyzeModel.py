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
from onnxruntime import InferenceSession


sess = InferenceSession('13_shape_conv.onnx')


def averageMiss(databaseFile, numOfSamples):
    test_fens, test_evals = Functions.read_database(databaseFile, numOfSamples)
    test_bitboards = Functions.process_multiple_fens_to_bit_board(test_fens)
    test_bitboards = test_bitboards.reshape(10000, 1, 13, 8, 8)
    modelEvalMiss = 0
    guessEvalMiss = 0
    for i in range(numOfSamples):
        prediction = sess.run(None, {'input': test_bitboards[0]})[0]
        modelEvalMiss += abs(test_evals[i] - prediction)
        guessEvalMiss += abs(test_evals[i] - 0)

    modelEvalAvgMiss = (modelEvalMiss / numOfSamples) * 1000
    guessEvalAvgMiss = (guessEvalMiss / numOfSamples) * 1000
    print("Average model miss - ", modelEvalAvgMiss, "Average random guess miss - ", guessEvalAvgMiss)



averageMiss("fens_evals_first_half.txt", 10000)
averageMiss("fens_evals_second_half.txt", 10000)



averageMiss("fens_evals_first_half.txt", 10000)
averageMiss("fens_evals_second_half.txt", 10000)