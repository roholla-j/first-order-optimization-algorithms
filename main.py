from optimizers.gd import gd
from optimizers.sgd import sgd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from utils.data import load_data
from sklearn.datasets import load_svmlight_file
from tests import logistic, rosenbrock, quadratic

OPTIMIZERS = {
    "GD":       (gd,       {}),
    "SGD":      (sgd,      {}),
    "Momentum": (momentum, {}),
    "NAG":      (nag,      {}),
    "Adam":     (adam,     {"eps": 1e-8}),
}

ALG_MENU = {
    "1": "GD",
    "3": "Momentum",
    "4": "NAG",
    "5": "Adam",
    "6": "Compare All",
}


def clear_screen():
    print("\033[H\033[J", end="")


def main():
    while True:
        clear_screen()
        print("=" * 50)
        print(" First-Order Optimization Algorithms Benchmark")
        print("=" * 50)

        print("\nChoose dataset/function:")
        print(" 1 - Logistic Regression (australian_scale.txt)")
        print(" 2 - Logistic Regression (rcv1_train.binary)")
        print(" 3 - Rosenbrock Function")
        print(" 4 - Quadratic Function (Bowl)")
        print(" q - Quit")
        dataset_choice = input("Enter choice [1-4, q]: ").strip().lower()
        if dataset_choice == "q":
            break

        if dataset_choice == "1":
            #X, y = load_svmlight_file("data/australian_scale.txt")
            X, y = load_data("data/australian_scale.txt")
            title_prefix = "Logistic (Australian)"
            n_iters = 200
        elif dataset_choice == "2":
            #X, y = load_svmlight_file("data/australian_scale.txt")
            X, y = load_data("data/rcv1_train.binary")
            title_prefix = "Logistic (RCV1)"
            n_iters = 100
        elif dataset_choice == "3":
            title_prefix = "Rosenbrock"
            n_iters = 2000
        elif dataset_choice == "4":
            title_prefix = "Quadratic Bowl"
            n_iters = 100
        else:
            print("Invalid choice.")
            continue

        print("\nChoose Algorithm:")
        print(" 1 - GD (Gradient Descent)")
        if dataset_choice in ["1", "2"]:
            print(" 2 - SGD (Stochastic Gradient Descent)")
        print(" 3 - Momentum")
        print(" 4 - NAG (Nesterov Accelerated Gradient)")
        print(" 5 - Adam")
        print(" 6 - Compare All")
        print(" q - Quit / Back to Dataset Selection")
        alg_choice = input("Enter choice: ").strip().lower()

        if alg_choice == "q":
            continue

        alg_map = {**ALG_MENU}
        if dataset_choice in ["1", "2"]:
            alg_map["2"] = "SGD"

        if alg_choice not in alg_map:
            print("Invalid or disabled choice.")
            continue

        alg_name = alg_map[alg_choice]
        print(f"\nRunning {alg_name} on {title_prefix}...\n")

        if dataset_choice in ["1", "2"]:
            logistic.run(alg_name, title_prefix, X, y, n_iters, OPTIMIZERS)
        elif dataset_choice == "3":
            rosenbrock.run(alg_name, n_iters, OPTIMIZERS)
        else:
            quadratic.run(alg_name, n_iters, OPTIMIZERS)


if __name__ == "__main__":
    main()
