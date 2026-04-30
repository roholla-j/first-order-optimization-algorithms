import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.gd import gd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from losses.quadratic import quadratic_loss, quadratic_gradient
from utils.plotting import make_axes_grid, plot_sensitivity_heatmap, plot_quadratic_paths
from metrics import lr_comparison, param_sweep, sensitivity

TITLE = "Quadratic Bowl"
W0    = np.array([3.0, 4.0])
A     = np.array([[3.0, 1.0], [1.0, 2.0]])
B     = np.array([2.0, 5.0])


def run(alg_name, n_iters, OPTIMIZERS):
    loss_func, gradient_func = quadratic_loss, quadratic_gradient
    config = {
        "n_iters": n_iters,
        "loss_func": loss_func,
        "gradient_func": gradient_func,
        "w0": W0,
        "w0_seed": 0,
    }

    if alg_name == "GD":
        _run_gd(config, OPTIMIZERS)

    elif alg_name in ["Momentum", "NAG", "Adam"]:
        opt_fn, base_kwargs = OPTIMIZERS[alg_name]
        _run_advanced(alg_name, opt_fn, base_kwargs, config)

    elif alg_name == "Compare All":
        _run_compare_all(n_iters, loss_func, gradient_func)


def _run_gd(config, OPTIMIZERS):
    config = {**config, "learning_rates": [1e-5, 1e-4, 5e-4, 1e-2]}
    results = lr_comparison.run({"GD": OPTIMIZERS["GD"]}, A, B, config)

    fig, axes = make_axes_grid(2)
    lr_comparison.plot(results, title=f"GD - Learning Rate Comparison ({TITLE})", ax=axes[0])

    paths = {f"lr={lr}": d["path"] for lr, d in results["GD"].items()}
    plot_quadratic_paths(paths, A=A, b=B, title="GD Trajectories (Quadratic Bowl)", ax=axes[1])

    plt.tight_layout()
    plt.show()


def _run_advanced(alg_name, opt_fn, base_kwargs, config):
    if alg_name == "Adam":
        LR_VALS        = [1e-4, 5e-4, 1e-3]
        FIXED_BETA1_LR = 0.9
        FIXED_BETA2_LR = 0.999
        BETA1_VALS     = [0.5, 0.9, 0.99]
        FIXED_LR_B1    = 5e-4
        FIXED_BETA2_B1 = 0.999
        BETA2_VALS     = [0.9, 0.99, 0.999]
        FIXED_LR_B2    = 5e-4
        FIXED_BETA1_B2 = 0.9

        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta1": FIXED_BETA1_LR, "beta2": FIXED_BETA2_LR},
            "lr", LR_VALS, A, B, config)
        print("  Running beta1 sweep...")
        beta1_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_B1, "beta2": FIXED_BETA2_B1},
            "beta1", BETA1_VALS, A, B, config)
        print("  Running beta2 sweep...")
        beta2_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_B2, "beta1": FIXED_BETA1_B2},
            "beta2", BETA2_VALS, A, B, config)

        fig, axes = make_axes_grid(4)
        param_sweep.plot(lr_res,    "lr",    f"beta1={FIXED_BETA1_LR}, beta2={FIXED_BETA2_LR}",
                         title=f"Adam — LR sweep ({TITLE})",    ax=axes[0])
        param_sweep.plot(beta1_res, "beta1", f"lr={FIXED_LR_B1}, beta2={FIXED_BETA2_B1}",
                         title=f"Adam — beta1 sweep ({TITLE})", ax=axes[1])
        param_sweep.plot(beta2_res, "beta2", f"lr={FIXED_LR_B2}, beta1={FIXED_BETA1_B2}",
                         title=f"Adam — beta2 sweep ({TITLE})", ax=axes[2])

        paths = {f"lr={lr}": data["path"] for lr, data in lr_res.items()}
        plot_quadratic_paths(paths, A=A, b=B, title="Adam Trajectories (Quadratic Bowl)", ax=axes[3])

    else:  # Momentum / NAG
        LR_VALS       = [1e-4, 5e-4, 1e-3]
        FIXED_BETA_LR = 0.9
        BETA_VALS     = [0.5, 0.9, 0.99]
        FIXED_LR_BETA = 5e-4

        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta": FIXED_BETA_LR},
            "lr", LR_VALS, A, B, config)
        print("  Running beta sweep...")
        beta_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_BETA},
            "beta", BETA_VALS, A, B, config)

        print("  Running sensitivity sweep...")
        sens_cfg = {**config, "param_grid": {
            "lr":   [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            "beta": [0.5, 0.7, 0.9, 0.95, 0.99],
        }}
        sens_results = sensitivity.run({alg_name: (opt_fn, base_kwargs)}, A, B, sens_cfg)

        fig, axes = make_axes_grid(4)
        param_sweep.plot(lr_res, "lr", f"beta={FIXED_BETA_LR}",
                         title=f"{alg_name} — LR sweep ({TITLE})", ax=axes[0])

        sens_data = sens_results[alg_name]
        plot_sensitivity_heatmap(
            sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
            row_label=sens_data["row_label"], col_label=sens_data["col_label"],
            title=f"lr × beta Sensitivity — {alg_name} ({TITLE})", ax=axes[1]
        )

        param_sweep.plot(beta_res, "beta", f"lr={FIXED_LR_BETA}",
                         title=f"{alg_name} — beta sweep ({TITLE})", ax=axes[2])

        paths = {f"lr={lr}": data["path"] for lr, data in lr_res.items()}
        plot_quadratic_paths(paths, A=A, b=B,
                             title=f"{alg_name} Trajectories (Quadratic Bowl)", ax=axes[3])

    plt.tight_layout()
    plt.show()


def _run_compare_all(n_iters, loss_func, gradient_func):
    best_optimizers = {
        "GD":       (gd,       {"lr": 1e-3}),
        "Momentum": (momentum, {"lr": 1e-3, "beta": 0.9}),
        "NAG":      (nag,      {"lr": 1e-3, "beta": 0.9}),
        "Adam":     (adam,     {"lr": 1e-2, "beta1": 0.9, "beta2": 0.999}),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    compare_paths = {}

    for i, (name, (opt_fn, kwargs)) in enumerate(tqdm(best_optimizers.items(), desc="Algorithms")):
        _, losses, path = opt_fn(
            start_w=W0,
            x=A,
            y=B,
            loss_func=loss_func,
            gradient_func=gradient_func,
            steps=n_iters,
            **kwargs,
        )
        ax1.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})",
                 color=colors[i % len(colors)], linewidth=2)
        compare_paths[name] = path

    ax1.set_title(f"Algorithm Comparison ({TITLE})", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    plot_quadratic_paths(compare_paths, A=A, b=B,
                         title="Algorithms Trajectories (Quadratic Bowl)", ax=ax2)

    plt.tight_layout()
    plt.show()
