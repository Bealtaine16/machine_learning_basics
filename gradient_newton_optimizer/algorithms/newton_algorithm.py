# algorithms/newton_algorithm.py
import numpy as np
from algorithms.gradient import GradientDescent
from config.config import (
    output,
    precision,
    n,
    epoch_size,
    step_condition_newton,
    decrease_newton,
)


class NewtonAlgorithm:
    def __init__(self):
        pass

    def derivative_hessian(self, x, h=0.1):
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

                y_plus_h_i_plus_h_j = output(x_plus_h_i + x_plus_h_j)
                y_minus_h_i_plus_h_j = output(x_minus_h_i + x_plus_h_j)
                y_plus_h_i_minus_h_j = output(x_plus_h_i + x_minus_h_j)
                y_minus_h_i_minus_h_j = output(x_minus_h_i + x_minus_h_j)

                hessian[i, j] = (
                    y_plus_h_i_plus_h_j
                    - y_minus_h_i_plus_h_j
                    - y_plus_h_i_minus_h_j
                    + y_minus_h_i_minus_h_j
                ) / (4 * h**2)

        return hessian

    def newton_algorithm(self, theta, learning_rate):
        gd_instance = GradientDescent()
        convergence_progress = []

        for epoch in range(epoch_size):
            is_decreased = True
            while is_decreased:
                is_decreased = False
                y = output(theta)
                gradient = gd_instance.derivative_gradient(theta)
                hessian = self.derivative_hessian(theta)
                hessian_inv = np.linalg.inv(hessian)
                d = np.dot(hessian_inv, gradient)
                theta = theta - learning_rate * d

                y_new = output(theta)
                if np.linalg.norm(gradient, 2) <= precision:
                    return convergence_progress

                # condition to adaptive learning rate
                # if y - y_new < step_condition_newton:
                #     learning_rate = learning_rate * decrease_newton
                #     is_decreased = True
                #     if learning_rate < step_condition_newton:
                #         print(
                #             f"[Gradient]Learn rate decreased to: {learning_rate}, function cannot find minimum"
                #         )
                #         break

            convergence_progress.append((epoch + 1, y_new))

        return convergence_progress
