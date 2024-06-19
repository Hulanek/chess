import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib as plt
from PIL import Image

# dataset from Tensorflow
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
ds_train = ds_train.cache()
ds_test = ds_test.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_test = ds_test.shuffle(ds_info.splits['test'].num_examples)
ds_train = ds_train.batch(10000)
ds_test = ds_test.batch(100)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = ds_train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
array = np.array(image_batch)
X = array.reshape(10000, 784)
y = np.array(labels_batch)
print(y)


class Layer:
    def __init__(self, numOfInputs, numOfNeurons):
        self.weights = np.random.randn(numOfInputs, numOfNeurons) * 0.1  # 0.1 so the generated number is small :D
        self.biases = np.zeros((1, numOfNeurons))  # setting default biases of each neuron to zeros

    def forward(self, inputs):
        self.inputs = inputs  # saving inputs for backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # the transposition change the array to state where rows contains weights that are connected to our layer
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:  # Rectified Linear Unit
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Loss:
     def calculate(self, output, y):
         sample_losses = self.forward(output, y)
         data_loss = np.mean(sample_losses)
         return data_loss


class Activation_Softmax:
     def forward(self, inputs):
         self.inputs = inputs
         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
         self.output = probabilities

     def backward(self, dvalues):
         self.dinputs = np.empty_like(dvalues)
         for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
             single_output = single_output.reshape(-1, 1)
             jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
             self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    # Initialize optimizer - set settings,
   # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0.):
           self.learning_rate = learning_rate
           self.current_learning_rate = learning_rate
           self.decay = decay
           self.iterations = 0
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
               self.current_learning_rate = self.learning_rate * \
                   (1. / (1. + self.decay * self.iterations))
    # Update parameters
    def update_params(self, layer):
           layer.weights += -self.current_learning_rate * layer.dweights
           layer.biases += -self.current_learning_rate * layer.dbiases
    # Call once after any parameter updates
    def post_update_params(self):
           self.iterations += 1


# Init
dense1 = Layer(784, 256)  # 28x28 pixels
activation1 = Activation_ReLU()
dense2 = Layer(256, 256)  # 28x28 pixels
activation2 = Activation_ReLU()
dense3 = Layer(256, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=1e-2)


for epoch in range(150):
    # Run
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    #if epoch % 100 == 0:
    print(f'epoch: {epoch}, ' +
          f'acc: {accuracy:.3f}, ' +
          f'loss: {loss:.3f}')

    # Backward
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update of weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


normalized_ds = ds_test.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
array = np.array(image_batch)
X = array.reshape(100, 784)
y = np.array(labels_batch)
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
loss = loss_activation.forward(dense3.output, y)
predictions = np.argmax(loss_activation.output, axis=1)
print(y)
print(predictions)
print(loss_activation.output)

'''
for number in range(10):
    strNumber = number
    img = Image.open('ownDigit_{}.png'.format(strNumber))
    rgbArray = np.array(img, dtype=np.float32)
    rgbArray = rgbArray.reshape(784, 3)
    myArray = np.zeros(784, float)

    for x in range(784):
        myArray[x] = (rgbArray[x][0] + rgbArray[x][1] + rgbArray[x][2]) / 3
    myArray = myArray / 255

    dense1.forward(myArray)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    print(strNumber, predictions, "    ", loss_activation.output)
'''