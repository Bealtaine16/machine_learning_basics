# main.py
import random
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from config.config import n, learning_rate, epoch_size
from algorithms.gradient import GradientDescent
from algorithms.newton_algorithm import NewtonAlgorithm


def main():
    random.seed(n)
    # theta = np.array([random.uniform(-100, 100) for i in range(n)], dtype=float)
    theta = np.array([76, -5, 96, 18, -43, -63, 54, 71, 48, 7], dtype=float)
    # theta = np.array(
    #     [
    #         -3,
    #         84,
    #         -11,
    #         -57,
    #         84,
    #         -15,
    #         56,
    #         69,
    #         70,
    #         -57,
    #         -28,
    #         -80,
    #         10,
    #         20,
    #         73,
    #         -28,
    #         -28,
    #         -44,
    #         -54,
    #         -19,
    #     ],
    #     dtype=float,
    # )

    # Gradient Descent
    start_dt_gradient = dt.datetime.now()
    gradient_instance = GradientDescent()
    conv_progress_gradient = gradient_instance.gradient_descent_algorithm(
        theta, learning_rate
    )
    last_epoch_gradient = conv_progress_gradient[-1][0]
    end_dt_gradient = dt.datetime.now()
    if last_epoch_gradient == epoch_size:
        annotation_gradient = f"The algorithm doesn't converged in {last_epoch_gradient} epochs.\nTime taken: {(end_dt_gradient - start_dt_gradient)}."
    else:
        annotation_gradient = f"The algorithm converged in {last_epoch_gradient} epochs.\nTime taken: {(end_dt_gradient - start_dt_gradient)}."

    # Newton Algorithm
    start_dt_newton = dt.datetime.now()
    newton_instance = NewtonAlgorithm()
    conv_progress_newton = newton_instance.newton_algorithm(theta, learning_rate)
    last_epoch_newton = conv_progress_newton[-1][0]
    end_dt_newton = dt.datetime.now()
    if last_epoch_newton == epoch_size:
        annotation_newton = f"The algorithm doesn't converged in {last_epoch_newton} epochs.\nTime taken: {(end_dt_newton - start_dt_newton)}."
    else:
        annotation_newton = f"The algorithm converged in {last_epoch_newton} epochs.\nTime taken: {(end_dt_newton - start_dt_newton)}."

    x_epoch_gradient, y_conv_gradient = zip(*conv_progress_gradient)
    x_epoch_newton, y_conv_newton = zip(*conv_progress_newton)

    # Create subplots
    plt.figure(figsize=(8, 8))

    # Subplot for Gradient Descent
    plt.subplot(2, 1, 1)
    plt.plot(x_epoch_gradient, y_conv_gradient, ".-", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Convergence Value")
    plt.title("Gradient Descent Convergence", fontsize=14, weight="bold")
    plt.annotate(
        annotation_gradient,
        xy=(1, 1),  # Coordinates (1, 1) are in the upper right corner of the axes
        xycoords="axes fraction",
        xytext=(-10, -10),  # Offset the text slightly
        textcoords="offset points",
        fontsize=10,
        ha="right",
        va="top",
    )

    # Subplot for Newton Algorithm
    plt.subplot(2, 1, 2)
    plt.plot(x_epoch_newton, y_conv_newton, ".-", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Convergence Value")
    plt.title("Newton Algorithm Convergence", fontsize=14, weight="bold")
    plt.annotate(
        annotation_newton,
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(-10, -10),
        textcoords="offset points",
        fontsize=10,
        ha="right",
        va="top",
    )

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


if __name__ == "__main__":
    main()
