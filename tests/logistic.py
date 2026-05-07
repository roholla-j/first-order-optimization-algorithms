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
from utils.config import get as cfg
from metrics import lr_comparison, sgd_comparison, param_sweep, sensitivity


def run(alg_name, title_prefix, X, y, n_iters, OPTIMIZERS, dataset_key):
    loss_func, gradient_func = logistic_loss, logistic_gradient
    config = {
        "n_iters": n_iters,
        "loss_func": loss_func,
        "gradient_func": gradient_func,
        "w0_seed": 0,
    }

    if alg_name == "GD":
        _run_gd(title_prefix, X, y, config, OPTIMIZERS, dataset_key)

    elif alg_name == "SGD":
        _run_sgd(title_prefix, X, y, n_iters, config, OPTIMIZERS, dataset_key)

    elif alg_name in ["Momentum", "NAG", "Adam"]:
        opt_fn, base_kwargs = OPTIMIZERS[alg_name]
        _run_advanced(alg_name, opt_fn, base_kwargs, title_prefix, X, y, config, dataset_key)

    elif alg_name == "Compare All":
        _run_compare_all(title_prefix, X, y, n_iters, loss_func, gradient_func, dataset_key)


def _run_gd(title_prefix, X, y, config, OPTIMIZERS, dataset_key):
    c = cfg()[dataset_key]["gd"]
    lr_config = {**config, "learning_rates": c["learning_rates"]}
    results = lr_comparison.run({"GD": OPTIMIZERS["GD"]}, X, y, lr_config)

    print("  Running batch size sweep...")
    fixed_lr = c["fixed_lr_for_batch"]
    opt_fn, base_kwargs = OPTIMIZERS["GD"]
    batch_config = {
        **config,
        "learning_rates": [fixed_lr],
        "batch_sizes": c["batch_sweep"],
        "n_epochs": config["n_iters"],
    }
    batch_results = sgd_comparison.run({"GD": (opt_fn, base_kwargs)}, X, y, batch_config)

    _, axes = make_axes_grid(2)
    lr_comparison.plot(results, title=f"GD — Learning Rate Comparison ({title_prefix})", ax=axes[0])
    sgd_comparison.plot_batch_sweep(
        batch_results, fixed_lr,
        title=f"GD — Batch Size Sweep (lr={fixed_lr}, {title_prefix})", ax=axes[1])
    plt.tight_layout()
    plt.show()


def _floor_pow2(n):
    return max(1, 1 << (n.bit_length() - 1))


def _run_sgd(title_prefix, X, y, n_iters, config, OPTIMIZERS, dataset_key):
    c = cfg()[dataset_key]["sgd"]
    N = X.shape[0]
    large = N > c["large_n_threshold"]
    max_epochs = c["n_epochs_large"] if large else n_iters
    if large:
        batch_sizes = [_floor_pow2(max(1, N // 100)), 512, _floor_pow2(max(1, N // 10)), _floor_pow2(max(1, N // 2))]
    else:
        batch_sizes = [1, 32, _floor_pow2(max(1, N // 10)), _floor_pow2(max(1, N // 2))]
    config = {**config,
              "n_epochs": max_epochs,
              "learning_rates": c["learning_rates"],
              "batch_sizes": batch_sizes}
    results = sgd_comparison.run({"SGD": OPTIMIZERS["SGD"]}, X, y, config)
    sgd_comparison.plot_by_batch_size(results, title=f"SGD — LR per Batch Size ({title_prefix})")


def _run_advanced(alg_name, opt_fn, base_kwargs, title_prefix, X, y, config, dataset_key):
    c = cfg()[dataset_key][alg_name.lower()]

    if alg_name == "Adam":
        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta1": c["fixed_beta1_for_lr"], "beta2": c["fixed_beta2_for_lr"]},
            "lr", c["lr_sweep"], X, y, config)
        print("  Running beta1 sweep...")
        beta1_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta1"], "beta2": c["fixed_beta2_for_beta1"]},
            "beta1", c["beta1_sweep"], X, y, config)
        print("  Running beta2 sweep...")
        beta2_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta2"], "beta1": c["fixed_beta1_for_beta2"]},
            "beta2", c["beta2_sweep"], X, y, config)

        print("  Running batch size sweep...")
        fixed_lr_batch = c["fixed_lr_for_batch"]
        batch_config = {
            **config,
            "learning_rates": [fixed_lr_batch],
            "batch_sizes": c["batch_sweep"],
            "n_epochs": config["n_iters"],
        }
        batch_results = sgd_comparison.run(
            {alg_name: (opt_fn, {**base_kwargs,
                                 "beta1": c["fixed_beta1_for_lr"],
                                 "beta2": c["fixed_beta2_for_lr"]})},
            X, y, batch_config)

        _, axes = make_axes_grid(4)
        param_sweep.plot(lr_res,    "lr",    f"beta1={c['fixed_beta1_for_lr']}, beta2={c['fixed_beta2_for_lr']}",
                         title=f"Adam — LR sweep ({title_prefix})",    ax=axes[0])
        param_sweep.plot(beta1_res, "beta1", f"lr={c['fixed_lr_for_beta1']}, beta2={c['fixed_beta2_for_beta1']}",
                         title=f"Adam — beta1 sweep ({title_prefix})", ax=axes[1])
        param_sweep.plot(beta2_res, "beta2", f"lr={c['fixed_lr_for_beta2']}, beta1={c['fixed_beta1_for_beta2']}",
                         title=f"Adam — beta2 sweep ({title_prefix})", ax=axes[2])
        sgd_comparison.plot_batch_sweep(
            batch_results, fixed_lr_batch,
            title=f"Adam — Batch Size Sweep (lr={fixed_lr_batch}, {title_prefix})",
            ax=axes[3])

    else:  # Momentum / NAG
        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta": c["fixed_beta_for_lr"]},
            "lr", c["lr_sweep"], X, y, config)
        print("  Running beta sweep...")
        beta_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta"]},
            "beta", c["beta_sweep"], X, y, config)

        print("  Running sensitivity sweep...")
        sens_cfg = {**config, "param_grid": {
            "lr":   c["sensitivity_lr_grid"],
            "beta": c["sensitivity_beta_grid"],
        }}
        sens_results = sensitivity.run({alg_name: (opt_fn, base_kwargs)}, X, y, sens_cfg)

        print("  Running batch size sweep...")
        fixed_lr_batch = c["fixed_lr_for_batch"]
        batch_config = {
            **config,
            "learning_rates": [fixed_lr_batch],
            "batch_sizes": c["batch_sweep"],
            "n_epochs": config["n_iters"],
        }
        batch_results = sgd_comparison.run(
            {alg_name: (opt_fn, {**base_kwargs, "beta": c["fixed_beta_for_lr"]})},
            X, y, batch_config)

        _, axes = make_axes_grid(4)
        param_sweep.plot(lr_res, "lr", f"beta={c['fixed_beta_for_lr']}",
                         title=f"{alg_name} — LR sweep ({title_prefix})", ax=axes[0])

        sens_data = sens_results[alg_name]
        plot_sensitivity_heatmap(
            sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
            row_label=sens_data["row_label"], col_label=sens_data["col_label"],
            title=f"lr × beta Sensitivity — {alg_name} ({title_prefix})", ax=axes[1]
        )

        param_sweep.plot(beta_res, "beta", f"lr={c['fixed_lr_for_beta']}",
                         title=f"{alg_name} — beta sweep ({title_prefix})", ax=axes[2])

        sgd_comparison.plot_batch_sweep(
            batch_results, fixed_lr_batch,
            title=f"{alg_name} — Batch Size Sweep (lr={fixed_lr_batch}, {title_prefix})",
            ax=axes[3])

    plt.tight_layout()
    plt.show()


def _run_compare_all(title_prefix, X, y, n_iters, loss_func, gradient_func, dataset_key):
    c = cfg()[dataset_key]["compare_all"]
    N = X.shape[0]

    best_optimizers = {
        "GD":       (gd,       {"lr": c["gd"]["lr"],       "batch_size": c["gd"]["batch_size"]}),
        "SGD":      (sgd,      {"lr": c["sgd"]["lr"],      "batch_size": c["sgd"]["batch_size"]}),
        "Momentum": (momentum, {"lr": c["momentum"]["lr"], "beta":  c["momentum"]["beta"],  "batch_size": c["momentum"]["batch_size"]}),
        "NAG":      (nag,      {"lr": c["nag"]["lr"],      "beta":  c["nag"]["beta"],       "batch_size": c["nag"]["batch_size"]}),
        "Adam":     (adam,     {"lr": c["adam"]["lr"],     "beta1": c["adam"]["beta1"],     "beta2": c["adam"]["beta2"], "batch_size": c["adam"]["batch_size"]}),
    }

    rng = np.random.default_rng(0)
    w0_init = rng.standard_normal(X.shape[1])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, (opt_fn, kwargs)) in enumerate(tqdm(best_optimizers.items(), desc="Algorithms")):
        spe = max(1, N // kwargs["batch_size"])
        _, losses, _ = opt_fn(
            start_w=w0_init,
            x=X,
            y=y,
            loss_func=loss_func,
            gradient_func=gradient_func,
            steps=n_iters * spe,
            **kwargs,
        )
        losses = losses[::spe][:n_iters]
        ax1.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})",
                 color=colors[i % len(colors)], linewidth=2)

    ax1.set_title(f"Algorithm Comparison ({title_prefix})", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
