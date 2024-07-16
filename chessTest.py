from keras import Sequential
from keras import layers
import tensorflow as tf
import tf2onnx
import Functions

fens, evals = Functions.read_database("fens_evals_first_half.txt", 150000)
input_bitboards = Functions.process_multiple_fens_to_bit_board(fens)

test_fens, test_evals = Functions.read_database("fens_evals_second_half.txt", 150000)
test_input_bitboards = Functions.process_multiple_fens_to_bit_board(test_fens)

evalModel = Sequential()
evalModel.add(layers.Conv2D(32, (3, 3), input_shape=(15, 8, 8), activation='relu'))
evalModel.add(layers.Conv2D(32, (3, 3), activation='relu'))
evalModel.add(layers.Flatten())
evalModel.add(layers.Dense(768, activation='relu'))
evalModel.add(layers.Dense(128, activation='relu'))
evalModel.add(layers.Dense(1, activation='tanh'))

evalModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
evalModel.fit(input_bitboards, evals, batch_size=1500, epochs=30, validation_data=(test_input_bitboards, test_evals))

# save as .keras model
# evalModel.save('model.keras')

input_spec = [tf.TensorSpec(shape=(1, 15, 8, 8), dtype=tf.float32, name='input')]
output = "new_bitboards_model.onnx"
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model=evalModel, input_signature=input_spec, output_path=output)




