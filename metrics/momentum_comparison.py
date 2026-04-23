"""
Momentum/NAG hyperparameter comparison metric
-----------------------------------------------
Sweeps learning_rates × betas for momentum-based optimizers and records the
full loss trajectory for every (lr, beta) pair.

Both lr and beta are injected by this module — do NOT include them in the
optimizer kwargs passed to run().
"""

import numpy as np
import matplotlib.pyplot as plt

from losses.logistic import logistic_loss, logistic_gradient

_LINESTYLES = ["-", "--", "-.", ":"]


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
                 The optimizer must accept `lr` and `beta` keyword arguments
                 (e.g. momentum, nag). Do NOT include `lr` or `beta` in kwargs.
    X, y       : training data
    config     : {
        learning_rates : list[float]   e.g. [0.01, 0.05, 0.1]
        betas          : list[float]   e.g. [0.5, 0.9, 0.99]
        n_iters        : int           number of iterations
        w0_seed        : int           random seed for initial weights
    }

    Returns
    -------
    {name: {lr: {beta: {"losses": list[float]}}}}
    """
    rng            = np.random.default_rng(config.get("w0_seed", 0))
    w0             = rng.standard_normal(X.shape[1])
    learning_rates = config["learning_rates"]
    betas          = config["betas"]
    n_iters        = config["n_iters"]
    results        = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        lr_results = {}
        for lr in learning_rates:
            beta_results = {}
            for beta in betas:
                _, losses, _ = opt_fn(
                    start_w=w0,
                    x=X,
                    y=y,
                    loss_func=logistic_loss,
                    gradient_func=logistic_gradient,
                    lr=lr,
                    beta=beta,
                    steps=n_iters,
                    **opt_kwargs,
                )
                beta_results[beta] = {"losses": losses}
            lr_results[lr] = beta_results
        results[name] = lr_results

    return results


def plot(results: dict, title="Momentum — lr × beta Comparison", save_path=None):
    """
    Parameters
    ----------
    results   : return value of run()
    title     : overall figure title
    save_path : if given, save figure to this path

    One subplot per algorithm. Within each subplot:
      - colors distinguish learning rates
      - linestyles distinguish beta values
    """
    colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_algs   = len(results)
    fig, axes = plt.subplots(1, n_algs, figsize=(9 * n_algs, 5), squeeze=False)

    for ax, (name, lr_dict) in zip(axes[0], results.items()):
        for i, (lr, beta_dict) in enumerate(lr_dict.items()):
            color = colors[i % len(colors)]
            for j, (beta, data) in enumerate(beta_dict.items()):
                ls    = _LINESTYLES[j % len(_LINESTYLES)]
                label = f"lr={lr},  β={beta}"
                ax.plot(data["losses"], label=label,
                        linewidth=1.8, color=color, linestyle=ls)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Logistic Loss", fontsize=12)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(title="lr  ·  β", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
