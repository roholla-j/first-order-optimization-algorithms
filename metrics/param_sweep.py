"""
Single-parameter sweep
-----------------------
Runs an optimizer with all hyperparameters fixed except one, which is
swept over a list of values.  Returns one loss trajectory per value.

Used by Momentum, NAG and Adam to draw the individual sweep line graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

_LINESTYLES = ["-", "--", "-."]
_COLORS     = None   # resolved lazily from rcParams


def run(opt_fn, fixed_kwargs, sweep_param, sweep_vals, X, y, config):
    """
    Parameters
    ----------
    opt_fn        : optimizer function
    fixed_kwargs  : dict of ALL hyperparameters that are held constant
                    (do NOT include sweep_param here)
    sweep_param   : name of the hyperparameter to vary (e.g. "lr", "beta")
    sweep_vals    : list of values to try
    X, y          : training data (may be None for analytical functions)
    config        : {
        n_iters       : int,
        loss_func     : callable,
        gradient_func : callable,
        w0            : np.ndarray  (used when X is None),
        w0_seed       : int
    }

    Returns
    -------
    {val: {"losses": list, "path": np.ndarray}}
    """
    rng  = np.random.default_rng(config.get("w0_seed", 0))
    w0   = rng.standard_normal(X.shape[1]) if X is not None else config.get("w0")
    # if config.get("w0") is not None:
    #     w0 = config["w0"].copy()
    # else:
    #     rng = np.random.default_rng(config.get("w0_seed", 0))
    #     # Fallback to random initialization for classification datasets
    #     w0 = rng.standard_normal(X.shape[1])
    n_iters       = config["n_iters"]
    loss_func     = config["loss_func"]
    gradient_func = config["gradient_func"]

    results = {}
    for val in tqdm(sweep_vals, desc=f"{sweep_param} sweep", leave=False):
        kwargs = {**fixed_kwargs, sweep_param: val}
        _, losses, path = opt_fn(
            start_w=w0,
            x=X,
            y=y,
            loss_func=loss_func,
            gradient_func=gradient_func,
            steps=n_iters,
            **kwargs,
        )
        results[val] = {"losses": losses, "path": path}

    return results


def plot(results, sweep_param, fixed_desc, title="", ax=None):
    """
    Parameters
    ----------
    results     : return value of run()
    sweep_param : name of the swept parameter (used in legend labels)
    fixed_desc  : human-readable string of the fixed values,
                  e.g. "beta=0.9"  or  "lr=0.1, beta2=0.999"
    title       : plot title
    ax          : matplotlib Axes to draw on; if None a new figure is created
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    show   = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(9, 5))

    for i, (val, data) in enumerate(results.items()):
        losses    = data["losses"]
        finite    = [v for v in losses if np.isfinite(v)]
        final_str = f"{finite[-1]:.4g}" if finite else "diverged"
        label     = f"{sweep_param}={val}  (Final: {final_str})"
        ax.plot(losses,
                label=label,
                color=colors[i % len(colors)],
                linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                linewidth=1.8)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title or f"{sweep_param} sweep", fontsize=12, fontweight="bold")
    ax.legend(title=f"Fixed: {fixed_desc}", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
