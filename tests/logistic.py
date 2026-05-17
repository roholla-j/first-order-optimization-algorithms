import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.gd import gd
from optimizers.sgd import sgd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from losses.logistic import logistic_loss, logistic_gradient
from utils.plotting import plots_dir, plot_sensitivity_heatmap
from utils.config import get as cfg
from metrics import lr_comparison, sgd_comparison, param_sweep, sensitivity


def _slug(s):
    return s.lower().replace("(", "").replace(")", "").replace(" ", "_").strip("_")


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")


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

    slug = _slug(title_prefix)
    d = plots_dir()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    lr_comparison.plot(results, title=f"GD — LR Comparison ({title_prefix})", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{slug}_gd_lr_comparison.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sgd_comparison.plot_batch_sweep(
        batch_results, fixed_lr,
        title=f"GD — Batch Sweep ({title_prefix})", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{slug}_gd_batch_sweep.pdf"))

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

    save_path = os.path.join(plots_dir(), f"{_slug(title_prefix)}_sgd_lr_per_batch_size.pdf")
    sgd_comparison.plot_by_batch_size(
        results, title=f"SGD — LR per Batch ({title_prefix})", save_path=save_path)
    print(f"  Saved: {save_path}")


def _run_advanced(alg_name, opt_fn, base_kwargs, title_prefix, X, y, config, dataset_key):
    c = cfg()[dataset_key][alg_name.lower()]
    slug = _slug(title_prefix)
    alg = alg_name.lower()
    d = plots_dir()

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

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(lr_res, "lr", f"beta1={c['fixed_beta1_for_lr']}, beta2={c['fixed_beta2_for_lr']}",
                         title=f"Adam — LR Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_lr_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta1_res, "beta1", f"lr={c['fixed_lr_for_beta1']}, beta2={c['fixed_beta2_for_beta1']}",
                         title=f"Adam — β₁ Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_beta1_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta2_res, "beta2", f"lr={c['fixed_lr_for_beta2']}, beta1={c['fixed_beta1_for_beta2']}",
                         title=f"Adam — β₂ Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_beta2_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sgd_comparison.plot_batch_sweep(
            batch_results, fixed_lr_batch,
            title=f"Adam — Batch Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_batch_sweep.pdf"))

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

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(lr_res, "lr", f"beta={c['fixed_beta_for_lr']}",
                         title=f"{alg_name} — LR Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_lr_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sens_data = sens_results[alg_name]
        plot_sensitivity_heatmap(
            sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
            row_label=sens_data["row_label"], col_label=sens_data["col_label"],
            title=f"{alg_name} — lr×β Sensitivity ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_sensitivity.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta_res, "beta", f"lr={c['fixed_lr_for_beta']}",
                         title=f"{alg_name} — β Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_beta_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sgd_comparison.plot_batch_sweep(
            batch_results, fixed_lr_batch,
            title=f"{alg_name} — Batch Sweep ({title_prefix})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{slug}_{alg}_batch_sweep.pdf"))

    plt.show()


def _run_compare_all(title_prefix, X, y, n_iters, loss_func, gradient_func, dataset_key):
    c = cfg()[dataset_key]["compare_all"]
    N = X.shape[0]
    bs = c["sgd"]["batch_size"]
    steps_per_epoch = max(1, N // bs)

    best_optimizers = {
        "GD":       (gd,       {"lr": c["gd"]["lr"]}),
        "SGD":      (sgd,      {"lr": c["sgd"]["lr"], "batch_size": bs}),
        "Momentum": (momentum, {"lr": c["momentum"]["lr"], "beta": c["momentum"]["beta"]}),
        "NAG":      (nag,      {"lr": c["nag"]["lr"],      "beta": c["nag"]["beta"]}),
        "Adam":     (adam,     {"lr": c["adam"]["lr"], "beta1": c["adam"]["beta1"], "beta2": c["adam"]["beta2"]}),
    }

    rng = np.random.default_rng(0)
    w0_init = rng.standard_normal(X.shape[1])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
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
        ax.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})",
                color=colors[i % len(colors)], linewidth=2)

    ax.set_title(f"Algorithm Comparison ({title_prefix})", fontweight="bold")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(plots_dir(), f"{_slug(title_prefix)}_compare_all.pdf"))
    plt.show()
