import random
import chess
import numpy as np
import Functions
import keras
from chess import pgn
import time
import tensorflow as tf


fen = "r1bqk1nr/ppp1n1bp/3p2p1/3Pp3/8/4P3/PPP1BPP1/RNBQK1NR w KQkq - 1 8"
fens = [100]
for i in range(0,len(fens)):
    fens[i] = fen
bitboards = Functions.process_multiple_fens_to_bit_board(fens)
bitboard = Functions.fen_to_bitboards(fen)
bitboard = bitboard.reshape(1, 15, 8, 8)

model = keras.models.load_model('firstModel.keras')

print("Durations using model __call__() on small batch")
for i in range(5):
    start = time.time()
    prediction = model(bitboard)
    end = time.time()
    print(end - start)

print("Durations using model.predict_on_batch() on small batch")
for i in range(5):
    start = time.time()
    prediction = model.predict_on_batch(bitboard)
    end = time.time()
    print(end - start)

print("Durations using model.predict() on small batch")
for i in range(5):
    start = time.time()
    prediction = model.predict(bitboard, batch_size=1, verbose=0)
    end = time.time()
    print(end - start)

print("Durations using model.__call__() on larger batch (100 samples)")
for i in range(5):
    start = time.time()
    prediction = model(bitboards)
    end = time.time()
    print(end - start)

print("Durations using model.predict_on_batch() on larger batch (100 samples)")
for i in range(5):
    start = time.time()
    prediction = model.predict_on_batch(bitboards)
    end = time.time()
    print(end - start)

print("Durations using model.predict() on larger batch (100 samples)")
for i in range(5):
    start = time.time()
    prediction = model.predict(bitboards, batch_size=1, verbose=0)
    end = time.time()
    print(end - start)
