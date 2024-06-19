import numpy as np
import keras
from keras import layers
from keras import models
from keras import utils
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin')
    return dict

import os
def load_batch_file(batch_filename):
    filepath = os.path.join('./cifar-10-batches-py', batch_filename)
    unpickled = unpickle(filepath)
    return unpickled

train_batch_1 = load_batch_file('data_batch_1')
train_batch_2 = load_batch_file('data_batch_2')
train_batch_3 = load_batch_file('data_batch_3')
train_batch_4 = load_batch_file('data_batch_4')
train_batch_5 = load_batch_file('data_batch_5')
test_batch = load_batch_file('test_batch')


num_classes = 10
train_x = np.concatenate([train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']])
train_x = train_x.astype('float32') # this is necessary for the division below
train_x /= 255
train_y = np.concatenate([utils.to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])

test_x = test_batch['data'].astype('float32') / 255
test_y = utils.to_categorical(test_batch['labels'], num_classes)






img_rows = img_cols = 32
channels = 3

train_x_reshaped = train_x.reshape(len(train_x), img_rows, img_cols, channels)
test_x_reshaped = test_x.reshape(len(test_x), img_rows, img_cols, channels)


simple_cnn_model = models.Sequential()
simple_cnn_model.add(layers.Conv2D(32, (3,3), input_shape=(img_rows,img_cols,channels), activation='relu'))
simple_cnn_model.add(layers.Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(layers.Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(layers.Flatten())
simple_cnn_model.add(layers.Dense(10, activation='softmax'))

simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn_model_history = simple_cnn_model.fit(train_x_reshaped, train_y, batch_size=100, epochs=8, validation_data=(test_x_reshaped, test_y))