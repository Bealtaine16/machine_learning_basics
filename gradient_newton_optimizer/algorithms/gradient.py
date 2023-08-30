# algorithms/gradient.py
import numpy as np
from config.config import learning_rate, output, precision


class GradientDescent:
    def __init__(self):
        pass

    def derivative_gradient(self, x, alfa, n, h=1e-05):
        gradient = np.zeros(n)

        for i in range(n):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            x_minus_h = x.copy()
            x_minus_h[i] -= h

            y_plus_h = output(x_plus_h, alfa, n)
            y_minus_h = output(x_minus_h, alfa, n)

            gradient[i] = (y_plus_h - y_minus_h) / (2 * h)

        return gradient

    def gradient_descent_algorithm(self, alfa, n, theta, epoch_size):
        cost_list = []
        mse_prev = None

        for epoch in range(epoch_size):
            y = output(theta, alfa, n)
            gradient = self.derivative_gradient(theta, alfa, n)
            theta = theta - learning_rate * gradient

            y_new = output(theta, alfa, n)
            mse = np.sum((y_new - y) ** 2)
            if mse_prev and abs(mse_prev - mse) <= precision:
                break
            mse_prev = mse
            cost_list.append((epoch, mse))

        return cost_list
