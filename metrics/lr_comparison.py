"""
Learning rate comparison metric
--------------------------------
Runs one or more optimizers with several learning rates (full-batch, no
mini-batching) and returns the loss trajectory for every (algorithm, lr) pair.
"""

import numpy as np
import matplotlib.pyplot as plt

from losses.logistic import logistic_loss, logistic_gradient
from tqdm import tqdm

# Linestyles cycled across learning rates so algorithms (colors) and
# learning rates (linestyles) remain visually separable on one graph.
_LINESTYLES = ["-", "--", "-.", ":"]


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
                 Matches the format used by all other metric modules.
                 e.g. {"GD": (gd, {}), "Momentum": (momentum, {"beta": 0.9})}
                 lr is injected automatically and must NOT appear in kwargs.
    X, y       : training data
    config     : {
        learning_rates : list[float],   e.g. [0.001, 0.01, 0.1, 1.0]
        n_iters        : int,           number of epochs per run
        w0_seed        : int,           random seed for initial weights
    }

    Returns
    -------
    {name: {lr: {"losses": list[float]}}}
    — same inner structure for every algorithm, one entry per lr.
    """
    w0             = np.zeros(X.shape[1]) if X is not None else config.get("w0")
    learning_rates = config["learning_rates"]
    n_iters        = config["n_iters"]
    loss_func      = config.get("loss_func", logistic_loss)
    gradient_func  = config.get("gradient_func", logistic_gradient)
    results        = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        lr_results = {}
        for lr in tqdm(learning_rates, desc=f"{name} LR sweep", leave=False):
            _, losses, path = opt_fn(
                start_w=w0,
                x=X,
                y=y,
                loss_func=loss_func,
                gradient_func=gradient_func,
                lr=lr,
                steps=n_iters,
                **opt_kwargs,
            )
            lr_results[lr] = {"losses": losses, "path": path}
        results[name] = lr_results

    return results


def plot(results: dict, title="Learning Rate Comparison", save_path=None, ax=None):
    """
    Parameters
    ----------
    results   : return value of run()
    title     : plot title
    save_path : if given, save figure to this path

    One curve per (algorithm, lr) pair.
    Learning rates are distinguished by color; algorithms by linestyle.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if ax is None:
        fig, ax_to_plot = plt.subplots(figsize=(10, 5))
        show = True
    else:
        ax_to_plot = ax
        show = False

    for i, (name, lr_dict) in enumerate(results.items()):
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        for j, (lr, data) in enumerate(lr_dict.items()):
            color = colors[j % len(colors)]
            base_label = f"{name}  lr={lr}" if len(results) > 1 else f"lr = {lr}"
            losses = data['losses']
            finite = [v for v in losses if np.isfinite(v)]
            final_str = f"{finite[-1]:.4g}" if finite else "diverged"
            label = f"{base_label} (Final: {final_str})"
            ax_to_plot.plot(losses, label=label,
                    linewidth=1.8, color=color, linestyle=ls)

    ax_to_plot.set_xlabel("Epoch / Iteration", fontsize=12)
    ax_to_plot.set_ylabel("Loss", fontsize=12)
    ax_to_plot.set_title(title, fontsize=12, fontweight="bold")
    legend_title = "Algorithm  ·  learning rate" if len(results) > 1 else "Learning rate"
    ax_to_plot.legend(title=legend_title, framealpha=0.9, fontsize=9)
    ax_to_plot.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
