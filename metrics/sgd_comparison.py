"""
SGD batch-size × learning-rate comparison metric
--------------------------------------------------
Sweeps learning_rates × batch_sizes for SGD-style optimizers and records the
full-data loss once per epoch so curves are epoch-normalised.

Epoch definition: one full pass over the data = ceil(N / batch_size) updates.
Both lr and batch_size are injected by this module — do NOT include them in the
optimizer kwargs passed to run().
"""

import numpy as np
import matplotlib.pyplot as plt

from losses.logistic import logistic_loss, logistic_gradient
from utils.plotting import make_axes_grid
from tqdm import tqdm

_LINESTYLES = ["-", "--", "-.", ":"]


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
                 The optimizer must accept `lr` and `batch_size` keyword args.
                 Do NOT include `lr` or `batch_size` in kwargs.
    X, y       : training data
    config     : {
        learning_rates : list[float]   e.g. [0.001, 0.01, 0.1, 1.0]
        batch_sizes    : list[int]     e.g. [1, 16, 64]
        n_epochs       : int           number of full passes over the data
        w0_seed        : int           random seed for initial weights
    }

    Returns
    -------
    {name: {lr: {batch_size: {"losses": list[float]}}}}
    losses has one value per epoch (full-data loss at each epoch boundary).
    """
    N              = X.shape[0] if X is not None else 1
    rng            = np.random.default_rng(config.get("w0_seed", 0))
    w0             = rng.standard_normal(X.shape[1]) if X is not None else config.get("w0")
    learning_rates = config["learning_rates"]
    batch_sizes    = config["batch_sizes"]
    n_epochs       = config["n_epochs"]
    loss_func      = config.get("loss_func", logistic_loss)
    gradient_func  = config.get("gradient_func", logistic_gradient)
    results        = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        lr_results = {}
        for lr in tqdm(learning_rates, desc=f"{name} LR sweep", leave=False):
            bs_results = {}
            for bs in batch_sizes:
                steps_per_epoch = max(1, N // bs)
                _, loss_history, _ = opt_fn(
                    start_w=w0,
                    x=X,
                    y=y,
                    loss_func=loss_func,
                    gradient_func=gradient_func,
                    lr=lr,
                    steps=n_epochs * steps_per_epoch,
                    batch_size=bs,
                    **opt_kwargs,
                )
                bs_results[bs] = {"losses": loss_history[::steps_per_epoch][:n_epochs]}
            lr_results[lr] = bs_results
        results[name] = lr_results

    return results


def plot(results: dict, title="SGD — lr × batch-size Comparison", save_path=None, ax=None):
    """Original combined plot (kept for backwards compatibility)."""
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_algs  = len(results)

    if ax is None:
        fig, axes = plt.subplots(1, n_algs, figsize=(9 * n_algs, 5), squeeze=False)
        axes_to_plot = axes[0]
        show = True
    else:
        axes_to_plot = [ax]
        show = False

    for ax_curr, (name, lr_dict) in zip(axes_to_plot, results.items()):
        for i, (lr, bs_dict) in enumerate(lr_dict.items()):
            color = colors[i % len(colors)]
            for j, (bs, data) in enumerate(bs_dict.items()):
                ls    = _LINESTYLES[j % len(_LINESTYLES)]
                losses = data["losses"]
                finite = [v for v in losses if np.isfinite(v)]
                final_str = f"{finite[-1]:.4g}" if finite else "diverged"
                label = f"lr={lr}, bs={bs} (Final: {final_str})"
                ax_curr.plot(losses, label=label,
                        linewidth=1.8, color=color, linestyle=ls)

        ax_curr.set_xlabel("Epoch", fontsize=12)
        ax_curr.set_ylabel("Loss", fontsize=12)
        ax_curr.set_title(name if show else title, fontsize=12, fontweight="bold")
        ax_curr.legend(title="lr  ·  batch size", framealpha=0.9, fontsize=8)
        ax_curr.grid(True, alpha=0.3)

    if show:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def plot_by_batch_size(results: dict, title="SGD — Batch Size Comparison", save_path=None):
    """
    One subplot per batch size; within each subplot, curves show different LRs.

    Parameters
    ----------
    results   : return value of run()  — {alg: {lr: {bs: {losses: [...]}}}}
    title     : overall figure super-title
    save_path : optional path to save the figure
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Collect batch sizes from the first algorithm / first lr
    name    = next(iter(results))
    lr_dict = results[name]
    first_bs_dict = next(iter(lr_dict.values()))
    batch_sizes   = list(first_bs_dict.keys())
    learning_rates = list(lr_dict.keys())

    n_bs = len(batch_sizes)
    fig, axes = make_axes_grid(n_bs)

    for ax, bs in zip(axes, batch_sizes):
        for i, lr in enumerate(learning_rates):
            data   = lr_dict[lr][bs]
            losses = data["losses"]
            finite = [v for v in losses if np.isfinite(v)]
            final_str = f"{finite[-1]:.4g}" if finite else "diverged"
            label  = f"lr={lr}  (Final: {final_str})"
            ax.plot(losses, label=label,
                    color=colors[i % len(colors)],
                    linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                    linewidth=1.8)

        ax.set_title(f"Batch size = {bs}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(title="Learning rate", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
