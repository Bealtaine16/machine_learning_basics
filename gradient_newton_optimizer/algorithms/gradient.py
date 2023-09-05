# algorithms/gradient.py
import numpy as np
from config.config import (
    output,
    precision,
    step_condition,
    decrease_gradient,
    epoch_size,
    alfa,
    n,
)


class GradientDescent:
    def __init__(self):
        pass

    def derivative_gradient(self, x, h=1e-05):
        gradient = np.zeros(n)

        for i in range(n):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            x_minus_h = x.copy()
            x_minus_h[i] -= h

            y_plus_h = output(x_plus_h)
            y_minus_h = output(x_minus_h)

            gradient[i] = (y_plus_h - y_minus_h) / (2 * h)

        return gradient

    def gradient_descent_algorithm(self, theta, learning_rate):
        convergence_progress = []
        y_new = 0

        for epoch in range(epoch_size):
            is_decreased = True
            while is_decreased:
                is_decreased = False
                y = output(theta)
                gradient = self.derivative_gradient(theta)
                theta = theta - learning_rate * gradient

                y_new = output(theta)
                if np.linalg.norm(gradient, 2) <= precision:
                    return convergence_progress

                # if y - y_new < step_condition:
                #     learning_rate = learning_rate * decrease_gradient
                #     is_decreased = True
                #     if learning_rate < step_condition:
                #         print(
                #             f"[Gradient]Learn rate decreased to: {learning_rate}, function cannot find minimum"
                #         )
                #         break

            convergence_progress.append((epoch + 1, y_new))

        return convergence_progress
