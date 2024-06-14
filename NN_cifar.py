import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin')
    return dict

import os
def load_batch_file(batch_filename):
    filepath = os.path.join('C:/Users/hulu0/Downloads/input/cifar-10-batches-py', batch_filename)
    unpickled = unpickle(filepath)
    return unpickled

train_batch_1 = load_batch_file('data_batch_1')
train_batch_2 = load_batch_file('data_batch_2')
train_batch_3 = load_batch_file('data_batch_3')
train_batch_4 = load_batch_file('data_batch_4')
train_batch_5 = load_batch_file('data_batch_5')
test_batch = load_batch_file('test_batch')

from keras.utils import to_categorical
num_classes = 10
train_x = np.concatenate([train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']])
train_x = train_x.astype('float32') # this is necessary for the division below
train_x /= 255
train_y = np.concatenate([to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])

test_x = test_batch['data'].astype('float32') / 255
test_y = to_categorical(test_batch['labels'], num_classes)

from keras.models import Sequential
from keras.layers import Dense

img_rows = img_cols = 32
channels = 3

simple_model = Sequential()
simple_model.add(Dense(10_000, input_shape=(img_rows*img_cols*channels,), activation='relu'))
#simple_model.add(Dense(1_000, activation='relu'))
#simple_model.add(Dense(100, activation='relu'))
simple_model.add(Dense(10, activation='softmax'))

simple_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model_history = simple_model.fit(train_x, train_y, batch_size=100, epochs=8, validation_data=(test_x, test_y))