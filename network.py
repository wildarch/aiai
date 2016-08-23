import numpy as np
import numpy.random as rand

from functools import reduce


class Network:
    def __init__(self, layer_sizes):
        # layer_sizes: list of numbers representing number of neurons per layer

        # Create a numpy array of biases for each layer except the (first) input layer
        self.biases = [rand.randn(l, 1) for l in layer_sizes[1:]]

        # The weights are an array of matrices. 'Between' each two layers is one matrix.
        # Every row contains a set of weights for each node
        self.weights = [rand.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feed_forward(self, input):
        # Perform a left fold
        return reduce(lambda input, b_w: np.dot(b_w[1], input)+b_w[0], zip(self.biases, self.weights), input)






def sigmoid(z):
    # The sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    # First-order derivative of the sigmoid function
    return sigmoid(z) * (1 - sigmoid(z))
