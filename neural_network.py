#!/usr/bin/env python

import numpy as np

from copy import copy

class NerualNetwork:
    MIN_LAYERS_NUM = 2
    MIN_LAYER_SIZE = 1

    def __init__(self):
        self._layer_sizes = [1, 1]
        self._weight_matrices = []
        self._init_weight_matrices()

    def init(self, layer_sizes):
        if len(layer_sizes) < self.MIN_LAYERS_NUM:
            print("Error: in NeuralNetwork.init len of layer_sizes is less than two")
            return False

        empty_layers = filter((lambda x: x < self.MIN_LAYER_SIZE), layer_sizes)
        if len(empty_layers) > 0:
            print("Error: in NeuralNetwork.init some of layer_sizes are equal to zero")
            return False

        self._layer_sizes = copy(layer_sizes)
        self._init_weight_matrices()
        return True

    def feed_forward(self, x):
        height, width = x.shape
        if height != self._layer_sizes[0]:
            print("Error: in NeuralNetwork.feed_forward size of input does not correspond to dimensions of the network")
            return None
 
        y_cap = copy(x)
        for w in self._weight_matrices:
            y_cap = np.dot(w, y_cap)
            y_cap = self._sigmoid(y_cap)

        return y_cap

    def add_gradients(self, x, y):
        gradients = self.get_gradients(x, y)
        for i, gradient in enumerate(gradients):
            self._weight_matrices[i] -= gradient

    def get_gradients(self, x, y):
        size_in, width_in = x.shape
        size_out, width_out = y.shape
        if width_in != width_out or \
                size_in != self._layer_sizes[0] or \
                size_out != self._layer_sizes[-1]:
            print("Error: in NeuralNetwork.get_gradient size of input does not correspond to dimensions of the network")
            return None

        num_layers = len(self._layer_sizes)
        z = [x]
        a = [x]
        s = [self._sigmoid_prime(x)]
        for w in self._weight_matrices:
            z.append(np.dot(w, a[-1]))
            a.append(self._sigmoid(z[-1]))
            s.append(self._sigmoid_prime(z[-1]))

        gradients = []
        # iteration on gradients for every of the weight matrices
        for j in range(num_layers - 1):
            # iteration on outputs of the networks
            gradient_average = np.zeros(self._weight_matrices[j].shape, np.float64)
            for i in range(size_out):
                y_current = y[i, :]
                y_cap_current = a[-1][i, :]

                gradient = -(y_current - y_cap_current)
                for k in range(1, num_layers - j - 1):
                    gradient = gradient * s[num_layers - k]
                    w_transposed = np.transpose(self._weight_matrices[num_layers - k - 1])
                    gradient = np.dot(w_transposed, gradient)
                gradient = gradient * s[j + 1]
                a_transposed = np.transpose(a[j])
                gradient = np.dot(gradient, a_transposed)
                gradient_average += gradient
            gradient_average /= size_out
            gradients.append(gradient_average)
        return gradients

    def _init_weight_matrices(self):
        self._weight_matrices = []
        for i in range(len(self._layer_sizes) - 1):
            size_current = self._layer_sizes[i]
            size_next = self._layer_sizes[i + 1]
            weight_matrix = np.random.rand(size_next, size_current)
            self._weight_matrices.append(weight_matrix)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        return np.exp(-x) / np.power(1 + np.exp(-x), 2)
