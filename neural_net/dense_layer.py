import numpy as np

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