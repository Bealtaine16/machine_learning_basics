import random
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

alfa_list = [1, 10, 100]
n_list = [10, 20]
learning_rate = 0.001
precision = 1e-05


def output(x, alfa, n):
    y = sum((alfa ** ((i - 1) / (n - 1))) * (x[i - 1] ** 2) for i in range(1, n + 1))
    return y


def output_predicted(x, theta, n):
    y = sum(theta[i - 1] * (x[i - 1] ** 2) for i in range(1, n + 1))
    return y


def derivative_mse(x, y_act, n, theta):
    dy = np.zeros(n)
    for i in range(n):
        dy[i] = 2 * (x[i] ** 2) * (theta[i] * (x[i] ** 2) - y_act)
    return dy


def gradient_descent_algorithm(alfa, n, epoch_size):
    random.seed(n)
    theta = np.array([random.uniform(-1, 1) for i in range(n)])
    x_inputs = np.array([random.uniform(-10, 10) for i in range(n)])
    y_act = output(x_inputs, alfa, n)
    cost_list = []
    mse_prev = None
    print(x_inputs, y_act, theta)

    for epoch in range(epoch_size):
        gradient = derivative_mse(x_inputs, y_act, n, theta)
        theta = theta - learning_rate * gradient
        mse = np.sum((output_predicted(x_inputs, theta, n) - y_act) ** 2)
        if mse_prev and abs(mse_prev - mse) <= precision:
            break
        mse_prev = mse
        print(y_act, output_predicted(x_inputs, theta, n), gradient)
        cost_list.append((epoch, mse))

    return cost_list


def main():
    start_dt = dt.datetime.now()

    cost = gradient_descent_algorithm(alfa_list[0], n_list[0], epoch_size=1000)

    end_dt = dt.datetime.now()
    print("Time taken: %s" % (end_dt - start_dt))

    x_epoch, y_cost = zip(*cost)

    plt.plot(x_epoch, y_cost, ".-", color="red")
    plt.show()


if __name__ == "__main__":
    main()

    # for epoch in range(epoch_size):

    #     act = [y for y in inputs(x_inputs, alfa, n)]
    #     print(act)
    #     grad = [gradient_descent(x, y, theta) for x, y in inputs(x_inputs, alfa, n)]
    #     mse = np.square(np.subtract(act, grad)).mean()
    #     grad_mean = np.mean(grad)
    #     theta = gradient_step(theta, grad, -learning_rate)
    #     if np.all(theta == [0, 0]):
    #         print(epoch)
    #         break
    #     cost_list.append((epoch, mse))

    # return cost_list

    #     y_pred = output(x_inputs, alfa, n)
    # mse = np.sum((y_act - y_pred) ** 2)
    # if mse and abs(mse_prev - mse) <= precision:
    #     break
    # mse_prev = mse
    # cost_list.append((epoch, mse))
