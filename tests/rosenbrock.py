import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.gd import gd
from optimizers.momentum import momentum
from optimizers.nag import nag
from optimizers.adam import adam

from losses.rosenbrock import rosenbrock, rosenbrock_gradient, distance_to_optimum
from utils.plotting import (plots_dir, plot_sensitivity_heatmap,
                            plot_rosenbrock_paths, plot_distance_to_optimum)
from utils.config import get as cfg
from metrics import lr_comparison, param_sweep, sensitivity

TITLE = "Rosenbrock"
_PREFIX = "rosenbrock"


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")


def run(alg_name, n_iters, OPTIMIZERS):
    c_top = cfg()["rosenbrock"]
    W0 = np.array(c_top["w0"])
    loss_func, gradient_func = rosenbrock, rosenbrock_gradient
    config = {
        "n_iters": n_iters,
        "loss_func": loss_func,
        "gradient_func": gradient_func,
        "w0": W0,
        "w0_seed": 0,
    }

    if alg_name == "GD":
        _run_gd(config, OPTIMIZERS, W0)

    elif alg_name in ["Momentum", "NAG", "Adam"]:
        opt_fn, base_kwargs = OPTIMIZERS[alg_name]
        _run_advanced(alg_name, opt_fn, base_kwargs, config, W0)

    elif alg_name == "Compare All":
        _run_compare_all(n_iters, loss_func, gradient_func, W0)


def _run_gd(config, OPTIMIZERS, W0):
    c = cfg()["rosenbrock"]["gd"]
    config = {**config, "learning_rates": c["learning_rates"]}
    results = lr_comparison.run({"GD": OPTIMIZERS["GD"]}, None, None, config)
    d = plots_dir()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    lr_comparison.plot(results, title=f"GD — LR Comparison ({TITLE})", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_gd_lr_comparison.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    paths = {f"lr={lr}": d_["path"] for lr, d_ in results["GD"].items()}
    plot_rosenbrock_paths(paths, title="GD Trajectories (Rosenbrock)", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_gd_trajectories.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    distances = {f"lr={lr}": distance_to_optimum(d_["path"]) for lr, d_ in results["GD"].items()}
    plot_distance_to_optimum(distances, title=f"GD — Distance to Optimum ({TITLE})", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_gd_distance_to_optimum.pdf"))

    plt.show()


def _run_advanced(alg_name, opt_fn, base_kwargs, config, W0):
    c = cfg()["rosenbrock"][alg_name.lower()]
    alg = alg_name.lower()
    d = plots_dir()

    if alg_name == "Adam":
        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta1": c["fixed_beta1_for_lr"], "beta2": c["fixed_beta2_for_lr"]},
            "lr", c["lr_sweep"], None, None, config)
        print("  Running beta1 sweep...")
        beta1_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta1"], "beta2": c["fixed_beta2_for_beta1"]},
            "beta1", c["beta1_sweep"], None, None, config)
        print("  Running beta2 sweep...")
        beta2_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta2"], "beta1": c["fixed_beta1_for_beta2"]},
            "beta2", c["beta2_sweep"], None, None, config)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(lr_res, "lr", f"beta1={c['fixed_beta1_for_lr']}, beta2={c['fixed_beta2_for_lr']}",
                         title=f"Adam — LR Sweep ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_lr_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta1_res, "beta1", f"lr={c['fixed_lr_for_beta1']}, beta2={c['fixed_beta2_for_beta1']}",
                         title=f"Adam — β₁ Sweep ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_beta1_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta2_res, "beta2", f"lr={c['fixed_lr_for_beta2']}, beta1={c['fixed_beta1_for_beta2']}",
                         title=f"Adam — β₂ Sweep ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_beta2_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        paths = {f"lr={lr}": data["path"] for lr, data in lr_res.items()}
        plot_rosenbrock_paths(paths, title="Adam Trajectories (Rosenbrock)", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_trajectories.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        distances = {f"lr={val}": distance_to_optimum(data["path"]) for val, data in lr_res.items()}
        plot_distance_to_optimum(distances, title=f"Adam — Distance to Optimum ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_distance_to_optimum.pdf"))

    else:  # Momentum / NAG
        print("  Running lr sweep...")
        lr_res = param_sweep.run(
            opt_fn, {**base_kwargs, "beta": c["fixed_beta_for_lr"]},
            "lr", c["lr_sweep"], None, None, config)
        print("  Running beta sweep...")
        beta_res = param_sweep.run(
            opt_fn, {**base_kwargs, "lr": c["fixed_lr_for_beta"]},
            "beta", c["beta_sweep"], None, None, config)

        print("  Running sensitivity sweep...")
        sens_cfg = {**config, "param_grid": {
            "lr":   c["sensitivity_lr_grid"],
            "beta": c["sensitivity_beta_grid"],
        }}
        sens_results = sensitivity.run({alg_name: (opt_fn, base_kwargs)}, None, None, sens_cfg)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(lr_res, "lr", f"beta={c['fixed_beta_for_lr']}",
                         title=f"{alg_name} — LR Sweep ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_lr_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sens_data = sens_results[alg_name]
        plot_sensitivity_heatmap(
            sens_data["grid"], sens_data["row_vals"], sens_data["col_vals"],
            row_label=sens_data["row_label"], col_label=sens_data["col_label"],
            title=f"{alg_name} — lr×β Sensitivity ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_sensitivity.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        param_sweep.plot(beta_res, "beta", f"lr={c['fixed_lr_for_beta']}",
                         title=f"{alg_name} — β Sweep ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_beta_sweep.pdf"))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        distances = {f"lr={val}": distance_to_optimum(data["path"]) for val, data in lr_res.items()}
        plot_distance_to_optimum(distances, title=f"{alg_name} — Distance to Optimum ({TITLE})", ax=ax)
        plt.tight_layout()
        _save(fig, os.path.join(d, f"{_PREFIX}_{alg}_distance_to_optimum.pdf"))

    plt.show()


def _run_compare_all(n_iters, loss_func, gradient_func, W0):
    c = cfg()["rosenbrock"]["compare_all"]
    best_optimizers = {
        "GD":       (gd,       {"lr": c["gd"]["lr"]}),
        "Momentum": (momentum, {"lr": c["momentum"]["lr"], "beta": c["momentum"]["beta"]}),
        "NAG":      (nag,      {"lr": c["nag"]["lr"], "beta": c["nag"]["beta"]}),
        "Adam":     (adam,     {"lr": c["adam"]["lr"], "beta1": c["adam"]["beta1"], "beta2": c["adam"]["beta2"]}),
    }

    compare_losses    = {}
    compare_paths     = {}
    compare_distances = {}

    for name, (opt_fn, kwargs) in tqdm(best_optimizers.items(), desc="Algorithms"):
        _, losses, path = opt_fn(
            start_w=W0,
            x=None,
            y=None,
            loss_func=loss_func,
            gradient_func=gradient_func,
            steps=n_iters,
            **kwargs,
        )
        compare_losses[name]    = losses
        compare_paths[name]     = path
        compare_distances[name] = distance_to_optimum(path)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    d = plots_dir()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, losses) in enumerate(compare_losses.items()):
        ax.plot(losses, label=f"{name} (Final Loss: {losses[-1]:.6f})",
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title(f"Algorithm Comparison ({TITLE})", fontweight="bold")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_compare_all_loss.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_rosenbrock_paths(compare_paths, title="Trajectories (Rosenbrock)", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_compare_all_trajectories.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_distance_to_optimum(compare_distances,
                             title="Distance to Optimum (Rosenbrock)", ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(d, f"{_PREFIX}_compare_all_distance_to_optimum.pdf"))

    plt.show()
