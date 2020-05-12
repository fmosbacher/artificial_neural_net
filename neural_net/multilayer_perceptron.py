import numpy as np

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

    # TODO: stop iteration with mse precision
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
                local_gradient = error * layer.activation(layer.outputs, deriv=True)

                # Update weights
                layer.weights += learning_rate * np.dot(layer.inputs.T, local_gradient)

                # Propagate error for next layers
                error = np.dot(local_gradient, layer.weights.T)