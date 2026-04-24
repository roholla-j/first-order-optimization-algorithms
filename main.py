import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.gd import gd
from optimizers.sgd import sgd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from losses.rosenbrock import rosenbrock, rosenbrock_gradient
from losses.quadratic import quadratic_loss, quadratic_gradient
from losses.logistic import logistic_loss, logistic_gradient

from utils.data import load_data
from utils.plotting import plot_sensitivity_heatmap, make_axes_grid
from metrics import lr_comparison, sgd_comparison, hyperparam_comparison, sensitivity, param_sweep

# --- Optimizers Config ---
OPTIMIZERS = {
    "GD": (gd, {}),
    "SGD": (sgd, {}),
    "Momentum": (momentum, {}),
    "NAG": (nag, {}),
    "Adam": (adam, {"eps": 1e-8}),
}

def clear_screen():
    print("\033[H\033[J", end="")

def main():
    while True:
        clear_screen()
        print("=" * 50)
        print(" First-Order Optimization Algorithms Benchmark")
        print("=" * 50)

        # 1. Choose Dataset / Function
        print("\nChoose dataset/function:")
        print(" 1 - Logistic Regression (australian_scale.txt)")
        print(" 2 - Logistic Regression (rcv1_train.binary)")
        print(" 3 - Rosenbrock Function")
        print(" 4 - Quadratic Function (Bowl)")
        print(" q - Quit")
        dataset_choice = input("Enter choice [1-4, q]: ").strip().lower()
        if dataset_choice == 'q':
            break

        X, y = None, None
        loss_func = None
        gradient_func = None
        w0 = None
        n_iters = 200

        if dataset_choice == "1":
            X, y = load_data("data/australian_scale.txt")
            loss_func, gradient_func = logistic_loss, logistic_gradient
            title_prefix = "Logistic (Australian)"
            n_iters = 200
        elif dataset_choice == "2":
            X, y = load_data("data/rcv1_train.binary")
            loss_func, gradient_func = logistic_loss, logistic_gradient
            title_prefix = "Logistic (RCV1)"
            n_iters = 200
        elif dataset_choice == "3":
            loss_func, gradient_func = rosenbrock, rosenbrock_gradient
            w0 = np.array([-1.0, 1.0])
            title_prefix = "Rosenbrock"
            n_iters = 4000
        elif dataset_choice == "4":
            loss_func, gradient_func = quadratic_loss, quadratic_gradient
            w0 = np.array([3.0, 4.0])
            # A and b for quadratic bowl
            X = np.array([[3.0, 1.0], [1.0, 2.0]])
            y = np.array([2.0, 5.0])
            title_prefix = "Quadratic Bowl"
            n_iters = 100
        else:
            print("Invalid choice.")
            continue

        # 2. Choose Algorithm
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

        if alg_choice == 'q':
            continue

        alg_name = ""
        if alg_choice == "1": alg_name = "GD"
        elif alg_choice == "2" and dataset_choice in ["1", "2"]: alg_name = "SGD"
        elif alg_choice == "3": alg_name = "Momentum"
        elif alg_choice == "4": alg_name = "NAG"
        elif alg_choice == "5": alg_name = "Adam"
        elif alg_choice == "6": alg_name = "Compare All"
        else:
            print("Invalid or disabled choice.")
            continue

        print(f"\nRunning {alg_name} on {title_prefix}...\n")

        # Common config defaults
        config = {
            "n_iters": n_iters,
            "loss_func": loss_func,
            "gradient_func": gradient_func,
            "w0": w0,
            "w0_seed": 0
        }

        if alg_name == "GD":
            config["learning_rates"] = [1e-3, 5e-3, 0.01, 0.1, 0.5, 1.0, 2.0] if dataset_choice in ["1","2"] else [1e-5, 1e-4, 5e-4, 1e-3]
            results = lr_comparison.run({"GD": OPTIMIZERS["GD"]}, X, y, config)

            if dataset_choice in ["3", "4"]:
                fig, axes = make_axes_grid(2)
                ax1, ax2  = axes
                lr_comparison.plot(results, title=f"GD - Learning Rate Comparison ({title_prefix})", ax=ax1)

                paths = {f"lr={lr}": d["path"] for lr, d in results["GD"].items()}
                if dataset_choice == "3":
                    from utils.plotting import plot_rosenbrock_paths
                    plot_rosenbrock_paths(paths, title="GD Trajectories (Rosenbrock)", ax=ax2)
                else:
                    from utils.plotting import plot_quadratic_paths
                    plot_quadratic_paths(paths, A=X, b=y, title="GD Trajectories (Quadratic Bowl)", ax=ax2)

                plt.tight_layout()
                plt.show()
            else:
                lr_comparison.plot(results, title=f"GD - Learning Rate Comparison ({title_prefix})")

        elif alg_name == "SGD":
            N = X.shape[0] if X is not None else 1
            # Cap epochs for large datasets (RCV1 ~20K samples — full sweep is slow)
            max_epochs = 50 if N > 10_000 else n_iters
            config["n_epochs"] = max_epochs
            config["learning_rates"] = [1e-3, 0.01, 0.1]
            # For very large N, skip batch_size=1 and start from N//100
            if N > 10_000:
                config["batch_sizes"] = [max(1, N // 100), max(1, N // 10), max(1, N // 2), N]
            else:
                config["batch_sizes"] = [1, max(1, N // 10), max(1, N // 2), N]
            results = sgd_comparison.run({"SGD": OPTIMIZERS["SGD"]}, X, y, config)
            sgd_comparison.plot_by_batch_size(results, title=f"SGD — LR per Batch Size ({title_prefix})")

        elif alg_name in ["Momentum", "NAG", "Adam"]:
            opt_fn, base_kwargs = OPTIMIZERS[alg_name]
            is_logistic = dataset_choice in ["1", "2"]

            # ----------------------------------------------------------------
            # Reference hyperparameter values — adjust these to change defaults
            # ----------------------------------------------------------------
            if alg_name == "Adam":
                # LR sweep: fixed beta1 and beta2
                LR_VALS        = [1e-3, 0.01, 0.1, 0.9]  if is_logistic else [1e-4, 5e-4, 1e-3]
                FIXED_BETA1_LR = 0.9
                FIXED_BETA2_LR = 0.999
                # beta1 sweep: fixed lr and beta2
                BETA1_VALS     = [0.5, 0.9, 0.99]
                FIXED_LR_B1    = 0.01  if is_logistic else 5e-4
                FIXED_BETA2_B1 = 0.999
                # beta2 sweep: fixed lr and beta1
                BETA2_VALS     = [0.9, 0.99, 0.999]
                FIXED_LR_B2    = 0.01  if is_logistic else 5e-4
                FIXED_BETA1_B2 = 0.9
            else:  # Momentum / NAG
                # LR sweep: fixed beta
                LR_VALS       = [1e-3, 0.01, 0.1, 0.9]  if is_logistic else [1e-4, 5e-4, 1e-3]
                FIXED_BETA_LR = 0.9
                # beta sweep: fixed lr
                BETA_VALS     = [0.5, 0.9, 0.99]
                FIXED_LR_BETA = 0.1    if is_logistic else 5e-4

            # ----------------------------------------------------------------
            # Run sweeps
            # ----------------------------------------------------------------
            if alg_name == "Adam":
                print(f"  Running lr sweep...")
                lr_res    = param_sweep.run(opt_fn, {**base_kwargs, "beta1": FIXED_BETA1_LR, "beta2": FIXED_BETA2_LR},
                                            "lr", LR_VALS, X, y, config)
                print(f"  Running beta1 sweep...")
                beta1_res = param_sweep.run(opt_fn, {**base_kwargs, "lr": FIXED_LR_B1, "beta2": FIXED_BETA2_B1},
                                            "beta1", BETA1_VALS, X, y, config)
                print(f"  Running beta2 sweep...")
                beta2_res = param_sweep.run(opt_fn, {**base_kwargs, "lr": FIXED_LR_B2, "beta1": FIXED_BETA1_B2},
                                            "beta2", BETA2_VALS, X, y, config)
            else:  # Momentum / NAG
                print(f"  Running lr sweep...")
                lr_res   = param_sweep.run(opt_fn, {**base_kwargs, "beta": FIXED_BETA_LR},
                                           "lr", LR_VALS, X, y, config)
                print(f"  Running beta sweep...")
                beta_res = param_sweep.run(opt_fn, {**base_kwargs, "lr": FIXED_LR_BETA},
                                           "beta", BETA_VALS, X, y, config)

                # Sensitivity heatmap (lr × beta grid)
                print(f"  Running sensitivity sweep...")
                if is_logistic:
                    lr_grid = [1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5]
                else:
                    lr_grid = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
                sens_cfg = config.copy()
                sens_cfg["param_grid"] = {
                    "lr":   lr_grid,
                    "beta": [0.5, 0.7, 0.9, 0.95, 0.99]
                }
                sens_results = sensitivity.run({alg_name: OPTIMIZERS[alg_name]}, X, y, sens_cfg)

            # ----------------------------------------------------------------
            # Layout and plotting
            # ----------------------------------------------------------------
            has_path_plot = dataset_choice in ["3", "4"]

            if alg_name == "Adam":
                # 3 line graphs + optional trajectory for analytical functions
                n_plots = 4 if has_path_plot else 3
                fig, axes = make_axes_grid(n_plots)
                param_sweep.plot(lr_res,    "lr",    f"beta1={FIXED_BETA1_LR}, beta2={FIXED_BETA2_LR}",
                                 title=f"Adam — LR sweep ({title_prefix})",    ax=axes[0])
                param_sweep.plot(beta1_res, "beta1", f"lr={FIXED_LR_B1}, beta2={FIXED_BETA2_B1}",
                                 title=f"Adam — beta1 sweep ({title_prefix})", ax=axes[1])
                param_sweep.plot(beta2_res, "beta2", f"lr={FIXED_LR_B2}, beta1={FIXED_BETA1_B2}",
                                 title=f"Adam — beta2 sweep ({title_prefix})", ax=axes[2])

                if has_path_plot:
                    paths = {f"lr={lr}": data["path"] for lr, data in lr_res.items()}
                    if dataset_choice == "3":
                        from utils.plotting import plot_rosenbrock_paths
                        plot_rosenbrock_paths(paths, title="Adam Trajectories (Rosenbrock)", ax=axes[3])
                    else:
                        from utils.plotting import plot_quadratic_paths
                        plot_quadratic_paths(paths, A=X, b=y, title="Adam Trajectories (Quadratic Bowl)", ax=axes[3])
            else:  # Momentum / NAG
                # 3 core graphs + optional trajectory
                n_plots = 4 if has_path_plot else 3
                fig, axes = make_axes_grid(n_plots)

                # Graph 1: LR sweep
                param_sweep.plot(lr_res, "lr", f"beta={FIXED_BETA_LR}",
                                 title=f"{alg_name} — LR sweep ({title_prefix})", ax=axes[0])

                # Graph 2: sensitivity heatmap
                sens_data = sens_results[alg_name]
                plot_sensitivity_heatmap(
                    sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
                    row_label=sens_data["row_label"], col_label=sens_data["col_label"],
                    title=f"lr × beta Sensitivity — {alg_name} ({title_prefix})", ax=axes[1]
                )

                # Graph 3: beta sweep
                param_sweep.plot(beta_res, "beta", f"lr={FIXED_LR_BETA}",
                                 title=f"{alg_name} — beta sweep ({title_prefix})", ax=axes[2])

                # Graph 4 (optional): trajectory paths
                if has_path_plot:
                    paths = {f"lr={lr}": data["path"] for lr, data in lr_res.items()}
                    if dataset_choice == "3":
                        from utils.plotting import plot_rosenbrock_paths
                        plot_rosenbrock_paths(paths,
                                             title=f"{alg_name} Trajectories (Rosenbrock)",
                                             ax=axes[3])
                    else:
                        from utils.plotting import plot_quadratic_paths
                        plot_quadratic_paths(paths, A=X, b=y,
                                             title=f"{alg_name} Trajectories (Quadratic Bowl)",
                                             ax=axes[3])

            plt.tight_layout()
            plt.show()


        elif alg_name == "Compare All":
            # Optimal params (rough estimates for comparison)
            best_optimizers = {
                "GD": (gd, {"lr": 0.1 if dataset_choice in ["1", "2"] else 1e-3}),
                "Momentum": (momentum, {"lr": 0.05 if dataset_choice in ["1", "2"] else 1e-3, "beta": 0.9}),
                "NAG": (nag, {"lr": 0.05 if dataset_choice in ["1", "2"] else 1e-3, "beta": 0.9}),
                "Adam": (adam, {"lr": 0.01 if dataset_choice in ["1", "2"] else 1e-2, "beta1": 0.9, "beta2": 0.999})
            }

            N = X.shape[0] if X is not None and len(X.shape) > 0 and dataset_choice in ["1", "2"] else 1

            if dataset_choice in ["1", "2"]:
                # SGD requires epoch normalization
                bs = max(1, N // 10)
                steps_per_epoch = max(1, N // bs)
                best_optimizers["SGD"] = (sgd, {"lr": 0.1, "batch_size": bs})

            rng = np.random.default_rng(0)
            w0_init = rng.standard_normal(X.shape[1]) if X is not None and dataset_choice in ["1", "2"] else w0

            if dataset_choice in ["3", "4"]:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(10, 6))

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            compare_paths = {}

            for i, (name, (opt_fn, kwargs)) in enumerate(tqdm(best_optimizers.items(), desc="Algorithms")):
                actual_steps = n_iters
                if name == "SGD" and dataset_choice in ["1", "2"]:
                    actual_steps = n_iters * steps_per_epoch

                _, losses, path = opt_fn(
                    start_w=w0_init,
                    x=X,
                    y=y,
                    loss_func=loss_func,
                    gradient_func=gradient_func,
                    steps=actual_steps,
                    **kwargs
                )

                if name == "SGD" and dataset_choice in ["1", "2"]:
                    # Epoch normalization
                    losses = losses[::steps_per_epoch][:n_iters]

                ax1.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})", color=colors[i % len(colors)], linewidth=2)

                if dataset_choice in ["3", "4"]:
                    compare_paths[name] = path

            ax1.set_title(f"Algorithm Comparison ({title_prefix})", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Epochs" if dataset_choice in ["1", "2"] else "Iterations", fontsize=12)
            ax1.set_ylabel("Loss", fontsize=12)
            ax1.legend(framealpha=0.9)
            ax1.grid(True, alpha=0.3)

            if dataset_choice == "3":
                from utils.plotting import plot_rosenbrock_paths
                plot_rosenbrock_paths(compare_paths, title="Algorithms Trajectories (Rosenbrock)", ax=ax2)
            elif dataset_choice == "4":
                from utils.plotting import plot_quadratic_paths
                plot_quadratic_paths(compare_paths, A=X, b=y, title="Algorithms Trajectories (Quadratic Bowl)", ax=ax2)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
