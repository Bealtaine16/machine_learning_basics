# config/config.py
import numpy as np


alfa_list = [1, 10, 100]
n_list = [10, 20]
precision = 0.01
learning_rate = 0.1
epoch_limit = 1000


def output(x, alfa, n):
    i = np.arange(1, n + 1)
    y = np.sum((alfa ** ((i - 1) / (n - 1))) * (x**2))
    return y
