# main.py
import random
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from config.config import alfa_list, n_list, epoch_limit
from algorithms.gradient import GradientDescent
from algorithms.newton_algorithm import NewtonAlgorithm


def main():
    random.seed(n_list[0])
    # theta = np.array([random.uniform(-100, 100) for i in range(n)])
    theta = np.array([76, -5, 96, 18, -43, -63, 54, 71, 48, 7])

    while True:
        algorithm_choice = input(
            "Which algorithm would you like to check? Indicate the number. (gradient (1)/newton (2)): "
        )

        if algorithm_choice == "1":
            start_dt = dt.datetime.now()
            gradient_instance = GradientDescent()
            cost = gradient_instance.gradient_descent_algorithm(
                alfa_list[0], n_list[0], theta, epoch_size=epoch_limit
            )
            last_epoch = cost[-1][0]
            print(f"The algorithm converged in {last_epoch} epochs.")
            end_dt = dt.datetime.now()
            chart_title = "Gradient Descent Convergence"
            break
        elif algorithm_choice == "2":
            start_dt = dt.datetime.now()
            newton_instance = NewtonAlgorithm()
            cost = newton_instance.newton_algorithm(
                alfa_list[0], n_list[0], theta, epoch_size=epoch_limit
            )
            last_epoch = cost[-1][0]
            print(f"The algorithm converged in {last_epoch} epochs.")
            end_dt = dt.datetime.now()
            chart_title = "Newton Algorithm Convergence"
            break
        elif algorithm_choice.lower() == "exit":
            print("Exiting the program.")
            sys.exit()
        else:
            print("Invalid choice. Please select 'gradient (1)' or 'newton (2)'.")

    print("Time taken: %s" % (end_dt - start_dt))

    x_epoch, y_cost = zip(*cost)

    plt.plot(x_epoch, y_cost, ".-", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(chart_title)
    plt.show()


if __name__ == "__main__":
    main()
