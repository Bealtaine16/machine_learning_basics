# config/config.py
import numpy as np


alfa = 1  # 10, 100
n = 10  # 20
precision = 0.01
learning_rate = 2.0  # 1  # 0.2  # 0.1
epoch_size = 1000
step_condition = 0.0001

decrease_gradient = 0.9
decrease_newton = 0.9


def output(x):
    i = np.arange(1, n + 1)
    y = np.sum((alfa ** ((i - 1) / (n - 1))) * (x**2))
    return y
