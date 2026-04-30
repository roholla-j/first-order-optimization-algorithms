import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.gd import gd
from optimizers.sgd import sgd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from losses.logistic import logistic_loss, logistic_gradient
from utils.plotting import make_axes_grid, plot_sensitivity_heatmap
from metrics import lr_comparison, sgd_comparison, param_sweep, sensitivity


def run(alg_name, title_prefix, X, y, n_iters, OPTIMIZERS):
    loss_func, gradient_func = logistic_loss, logistic_gradient
    config = {
        "n_iters": n_iters,
        "loss_func": loss_func,
        "gradient_func": gradient_func,
        "w0_seed": 0,
    }

    if alg_name == "GD":
        _run_gd(title_prefix, X, y, config, OPTIMIZERS)

    elif alg_name == "SGD":
        _run_sgd(title_prefix, X, y, n_iters, config, OPTIMIZERS)

    elif alg_name in ["Momentum", "NAG", "Adam"]:
        opt_fn, base_kwargs = OPTIMIZERS[alg_name]
        _run_advanced(alg_name, opt_fn, base_kwargs, title_prefix, X, y, config)

    elif alg_name == "Compare All":
        _run_compare_all(title_prefix, X, y, n_iters, loss_func, gradient_func)


def _run_gd(title_prefix, X, y, config, OPTIMIZERS):
    config = {**config, "learning_rates": [2, 4, 8, 12]}
    results = lr_comparison.run({"GD": OPTIMIZERS["GD"]}, X, y, config)
    lr_comparison.plot(results, title=f"GD - Learning Rate Comparison ({title_prefix})")


def _run_sgd(title_prefix, X, y, n_iters, config, OPTIMIZERS):
    N = X.shape[0]
    max_epochs = 50 if N > 10_000 else n_iters
    if N > 10_000:
        batch_sizes = [max(1, N // 100), max(1, N // 10), max(1, N // 2), N]
    else:
        batch_sizes = [1, max(1, N // 10), max(1, N // 2), N]
    config = {**config,
              "n_epochs": max_epochs,
              "learning_rates": [1e-3, 0.01, 0.1],
              "batch_sizes": batch_sizes}
    results = sgd_comparison.run({"SGD": OPTIMIZERS["SGD"]}, X, y, config)
    sgd_comparison.plot_by_batch_size(results, title=f"SGD — LR per Batch Size ({title_prefix})")


def _run_advanced(alg_name, opt_fn, base_kwargs, title_prefix, X, y, config):
    if alg_name == "Adam":
        LR_VALS        = [1e-3, 0.01, 0.1, 0.9]
        FIXED_BETA1_LR = 0.9
        FIXED_BETA2_LR = 0.999
        BETA1_VALS     = [0.5, 0.9, 0.99]
        FIXED_LR_B1    = 0.01
        FIXED_BETA2_B1 = 0.999
        BETA2_VALS     = [0.9, 0.99, 0.999]
        FIXED_LR_B2    = 0.01
        FIXED_BETA1_B2 = 0.9

        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta1": FIXED_BETA1_LR, "beta2": FIXED_BETA2_LR},
            "lr", LR_VALS, X, y, config)
        print("  Running beta1 sweep...")
        beta1_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_B1, "beta2": FIXED_BETA2_B1},
            "beta1", BETA1_VALS, X, y, config)
        print("  Running beta2 sweep...")
        beta2_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_B2, "beta1": FIXED_BETA1_B2},
            "beta2", BETA2_VALS, X, y, config)

        fig, axes = make_axes_grid(3)
        param_sweep.plot(lr_res,    "lr",    f"beta1={FIXED_BETA1_LR}, beta2={FIXED_BETA2_LR}",
                         title=f"Adam — LR sweep ({title_prefix})",    ax=axes[0])
        param_sweep.plot(beta1_res, "beta1", f"lr={FIXED_LR_B1}, beta2={FIXED_BETA2_B1}",
                         title=f"Adam — beta1 sweep ({title_prefix})", ax=axes[1])
        param_sweep.plot(beta2_res, "beta2", f"lr={FIXED_LR_B2}, beta1={FIXED_BETA1_B2}",
                         title=f"Adam — beta2 sweep ({title_prefix})", ax=axes[2])

    else:  # Momentum / NAG
        LR_VALS       = [1e-3, 0.01, 0.1, 0.9]
        FIXED_BETA_LR = 0.9
        BETA_VALS     = [0.5, 0.9, 0.99]
        FIXED_LR_BETA = 0.1

        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta": FIXED_BETA_LR},
            "lr", LR_VALS, X, y, config)
        print("  Running beta sweep...")
        beta_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": FIXED_LR_BETA},
            "beta", BETA_VALS, X, y, config)

        print("  Running sensitivity sweep...")
        sens_cfg = {**config, "param_grid": {
            "lr":   [1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5],
            "beta": [0.5, 0.7, 0.9, 0.95, 0.99],
        }}
        sens_results = sensitivity.run({alg_name: (opt_fn, base_kwargs)}, X, y, sens_cfg)

        fig, axes = make_axes_grid(3)
        param_sweep.plot(lr_res, "lr", f"beta={FIXED_BETA_LR}",
                         title=f"{alg_name} — LR sweep ({title_prefix})", ax=axes[0])

        sens_data = sens_results[alg_name]
        plot_sensitivity_heatmap(
            sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
            row_label=sens_data["row_label"], col_label=sens_data["col_label"],
            title=f"lr × beta Sensitivity — {alg_name} ({title_prefix})", ax=axes[1]
        )

        param_sweep.plot(beta_res, "beta", f"lr={FIXED_LR_BETA}",
                         title=f"{alg_name} — beta sweep ({title_prefix})", ax=axes[2])

    plt.tight_layout()
    plt.show()


def _run_compare_all(title_prefix, X, y, n_iters, loss_func, gradient_func):
    N = X.shape[0]
    bs = max(1, N // 10)
    steps_per_epoch = max(1, N // bs)

    best_optimizers = {
        "GD":       (gd,       {"lr": 0.1}),
        "SGD":      (sgd,      {"lr": 0.1, "batch_size": bs}),
        "Momentum": (momentum, {"lr": 0.05, "beta": 0.9}),
        "NAG":      (nag,      {"lr": 0.05, "beta": 0.9}),
        "Adam":     (adam,     {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
    }

    rng = np.random.default_rng(0)
    w0_init = rng.standard_normal(X.shape[1])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, (opt_fn, kwargs)) in enumerate(tqdm(best_optimizers.items(), desc="Algorithms")):
        actual_steps = n_iters * steps_per_epoch if name == "SGD" else n_iters
        _, losses, _ = opt_fn(
            start_w=w0_init,
            x=X,
            y=y,
            loss_func=loss_func,
            gradient_func=gradient_func,
            steps=actual_steps,
            **kwargs,
        )
        if name == "SGD":
            losses = losses[::steps_per_epoch][:n_iters]
        ax1.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})",
                 color=colors[i % len(colors)], linewidth=2)

    ax1.set_title(f"Algorithm Comparison ({title_prefix})", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
