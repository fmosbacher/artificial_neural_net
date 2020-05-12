import numpy as np
import math

def relu(values, deriv=False):
    """Leaky ReLU implementation with derivative for activation function."""
    alpha = 0.1

    if deriv:
        return np.where(values > 0, 1, alpha)

    return np.where(values > 0, values, values * alpha)

class DenseLayer():
    """Dense neural net layer to build the MLP."""

    def __init__(self, n_inputs, n_outputs, activation=relu):
        self.weights = np.random.random((n_inputs, n_outputs)) * 2 - 1
        self.bias = np.random.random((1, n_outputs))
        self.inputs = None
        self.outputs = None
        self.activated_outputs = None
        self.activation = activation

    def feed_forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.bias
        self.activated_outputs = self.activation(self.outputs)

class MultilayerPerceptron:
    """Simple neural net implementation."""

    def __init__(self, layers_shape):
        self.layers = []
        for i in range(len(layers_shape) - 1):
            input_shape = layers_shape[i]
            output_shape = layers_shape[i + 1]
            self.layers.append(DenseLayer(input_shape, output_shape))

    def predict(self, inputs):
        next_inputs = np.array(inputs)

        for layer in self.layers:
            layer.feed_forward(next_inputs)
            next_inputs = layer.activated_outputs

        return self.layers[-1].activated_outputs

    def train(self, inputs_batch, targets_batch, learning_rate=0.01, precision=0.0001):
        for i in range(10000):
            # Feedforward
            next_inputs = np.array(inputs_batch)
            for layer in self.layers:
                layer.feed_forward(next_inputs)
                next_inputs = layer.activated_outputs

            # Backpropagation
            error = targets_batch - self.layers[-1].activated_outputs

            if i % 1000 == 0:
                print('> Mean squared error:', np.mean(error ** 2))

            for layer in reversed(self.layers):
                local_gradient = error * relu(layer.outputs, deriv=True)

                # Update weights
                layer.weights += learning_rate * np.dot(layer.inputs.T, local_gradient)

                # Propagate error for next layers
                error = np.dot(local_gradient, layer.weights.T)

# np.random.seed()

shape = [2, 8, 4, 8, 4, 8, 4, 1]

mlp = MultilayerPerceptron(shape)

inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

targets = [[0],
           [1],
           [1],
           [0]]

mlp.train(inputs, targets)

print('> Used shape: ', shape)

print('> [0, 1]: ', mlp.predict([0, 1]))
print('> [1, 1]: ', mlp.predict([1, 1]))