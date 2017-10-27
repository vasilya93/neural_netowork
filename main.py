#!/usr/bin/env python

import numpy as np

from neural_network import NerualNetwork

def himmelblau(x):
    x1 = x[0, :]
    x2 = x[1, :]
    _, width = x.shape
    first_term = np.power(np.power(x1, 2) + x2 - 11, 2)
    second_term = np.power(np.power(x2, 2) + x1 - 7, 2)
    y = first_term + second_term
    y = np.reshape(y, (1, width))
    return y

def parabola(x):
    x1 = x[0, :]
    x2 = x[1, :]
    _, width = x.shape
    first_term = np.power(x1 - 3, 2)
    second_term = np.power(x2 + 6, 2)
    y = first_term + second_term
    y = np.reshape(y, (1, width))
    return y

nn = NerualNetwork()
nn.init([2, 10, 1])

num_samples = 100
x = (np.random.rand(2, num_samples) - 0.5) * 12
y = parabola(x)

x_scaled = x / np.max(x)
y_scaled = y / np.max(y)

y_cap = nn.feed_forward(x_scaled)
average_cost = np.sum(np.power((y_scaled - y_cap), 2)) / num_samples
print("cost 0: %f" % average_cost)
for i in range(1, 101):
    nn.add_gradients(x_scaled, y_scaled)
    y_cap = nn.feed_forward(x_scaled)
    average_cost = np.sum(np.power((y_scaled - y_cap), 2)) / num_samples
    print("cost %d: %f" % (i, average_cost))
