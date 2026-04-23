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
    N              = X.shape[0]
    rng            = np.random.default_rng(config.get("w0_seed", 0))
    w0             = rng.standard_normal(X.shape[1])
    learning_rates = config["learning_rates"]
    batch_sizes    = config["batch_sizes"]
    n_epochs       = config["n_epochs"]
    results        = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        lr_results = {}
        for lr in learning_rates:
            bs_results = {}
            for bs in batch_sizes:
                steps_per_epoch = max(1, N // bs)
                _, loss_history, _ = opt_fn(
                    start_w=w0,
                    x=X,
                    y=y,
                    loss_func=logistic_loss,
                    gradient_func=logistic_gradient,
                    lr=lr,
                    steps=n_epochs * steps_per_epoch,
                    batch_size=bs,
                    **opt_kwargs,
                )
                bs_results[bs] = {"losses": loss_history[::steps_per_epoch][:n_epochs]}
            lr_results[lr] = bs_results
        results[name] = lr_results

    return results


def plot(results: dict, title="SGD — lr × batch-size Comparison", save_path=None):
    """
    Parameters
    ----------
    results   : return value of run()
    title     : overall figure title
    save_path : if given, save figure to this path

    One subplot per algorithm. Within each subplot:
      - colors distinguish learning rates
      - linestyles distinguish batch sizes
    """
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_algs  = len(results)
    fig, axes = plt.subplots(1, n_algs, figsize=(9 * n_algs, 5), squeeze=False)

    for ax, (name, lr_dict) in zip(axes[0], results.items()):
        for i, (lr, bs_dict) in enumerate(lr_dict.items()):
            color = colors[i % len(colors)]
            for j, (bs, data) in enumerate(bs_dict.items()):
                ls    = _LINESTYLES[j % len(_LINESTYLES)]
                label = f"lr={lr},  bs={bs}"
                ax.plot(data["losses"], label=label,
                        linewidth=1.8, color=color, linestyle=ls)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Logistic Loss", fontsize=12)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(title="lr  ·  batch size", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
