# algorithms/newton_algorithm.py
import numpy as np
from algorithms.gradient import GradientDescent
from config.config import learning_rate, output, precision


class NewtonAlgorithm:
    def __init__(self):
        pass

    def derivative_hessian(self, x, alfa, n, h=0.1):
        hessian = np.zeros((n, n))

        for i in range(n):
            x_plus_h_i = x.copy()
            x_plus_h_i[i] += h
            x_minus_h_i = x.copy()
            x_minus_h_i[i] -= h
            for j in range(n):
                x_plus_h_j = x.copy()
                x_plus_h_j[j] += h
                x_minus_h_j = x.copy()
                x_minus_h_j[j] -= h

                y_plus_h_i_plus_h_j = output(x_plus_h_i + x_plus_h_j, alfa, n)
                y_minus_h_i_plus_h_j = output(x_minus_h_i + x_plus_h_j, alfa, n)
                y_plus_h_i_minus_h_j = output(x_plus_h_i + x_minus_h_j, alfa, n)
                y_minus_h_i_minus_h_j = output(x_minus_h_i + x_minus_h_j, alfa, n)

                hessian[i, j] = (
                    y_plus_h_i_plus_h_j
                    - y_minus_h_i_plus_h_j
                    - y_plus_h_i_minus_h_j
                    + y_minus_h_i_minus_h_j
                ) / (4 * h**2)

        return hessian

    def newton_algorithm(self, alfa, n, theta, epoch_size):
        gd_instance = GradientDescent()
        cost_list = []
        mse_prev = None

        for epoch in range(epoch_size):
            y = output(theta, alfa, n)
            gradient = gd_instance.derivative_gradient(theta, alfa, n)
            hessian = self.derivative_hessian(theta, alfa, n)
            hessian_inv = np.linalg.inv(hessian)
            d = np.dot(hessian_inv, gradient)
            theta = theta - learning_rate * d

            y_new = output(theta, alfa, n)
            mse = np.sum((y_new - y) ** 2)
            if mse_prev and abs(mse_prev - mse) <= precision:
                break
            mse_prev = mse
            cost_list.append((epoch, mse))

        return cost_list
