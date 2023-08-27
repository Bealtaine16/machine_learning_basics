import random
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# Constants
alfa_list = [1, 10, 100]
n_list = [10, 20]
precision = 0.0001  # 1e-05
learning_rate = 0.1  # 0.01
epoch_limit = 1000


def output(x, alfa, n):
    i = np.arange(1, n + 1)
    y = np.sum((alfa ** ((i - 1) / (n - 1))) * (x**2))
    return y


def derivative(x, alfa, n):
    i = np.arange(1, n + 1)
    dy = (2 * alfa ** ((i - 1) / (n - 1))) * x
    return dy


def derivative_second(x, alfa, n):
    i = np.arange(1, n + 1)
    d2y = 2 * alfa ** ((i - 1) / (n - 1))
    return d2y


def gradient_descent_algorithm(alfa, n, epoch_size):
    random.seed(n)
    # theta = np.array([random.uniform(-100, 100) for i in range(n)])
    theta = np.array([76, -5, 96, 18, -43, -63, 54, 71, 48, 7])
    cost_list = []
    mse_prev = None

    for epoch in range(epoch_size):
        y = output(theta, alfa, n)
        gradient = derivative(theta, alfa, n)
        theta = theta - learning_rate * gradient

        y_new = output(theta, alfa, n)
        mse = np.sum((y_new - y) ** 2)
        if mse_prev and abs(mse_prev - mse) <= precision:
            break
        mse_prev = mse
        cost_list.append((epoch, mse))
        print(epoch, mse)

    return cost_list


def main():
    start_dt = dt.datetime.now()

    cost = gradient_descent_algorithm(alfa_list[0], n_list[0], epoch_size=epoch_limit)

    end_dt = dt.datetime.now()
    print("Time taken: %s" % (end_dt - start_dt))

    x_epoch, y_cost = zip(*cost)

    plt.plot(x_epoch, y_cost, ".-", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Gradient Descent Convergence")
    plt.show()


if __name__ == "__main__":
    main()
