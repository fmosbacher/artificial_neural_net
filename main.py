from neural_net import MultilayerPerceptron

shape = [2, 4, 1]

mlp = MultilayerPerceptron(shape)

# XOR example

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