import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

_LINESTYLES = ["-", "--", "-.", ":"]

def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Sweeps param1 × param2 for optimizers and records full loss trajectory.
    """
    rng = np.random.default_rng(config.get("w0_seed", 0))
    w0 = rng.standard_normal(X.shape[1]) if X is not None else config.get("w0")
    
    p1_name = config["param1_name"]
    p2_name = config["param2_name"]
    p1_vals = config["param1_vals"]
    p2_vals = config["param2_vals"]
    n_iters = config["n_iters"]
    
    loss_func = config["loss_func"]
    gradient_func = config["gradient_func"]
    
    results = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        p1_results = {}
        for v1 in tqdm(p1_vals, desc=f"{name} {p1_name} sweep", leave=False):
            p2_results = {}
            for v2 in p2_vals:
                kwargs = opt_kwargs.copy()
                kwargs[p1_name] = v1
                kwargs[p2_name] = v2
                
                _, losses, path = opt_fn(
                    start_w=w0,
                    x=X,
                    y=y,
                    loss_func=loss_func,
                    gradient_func=gradient_func,
                    steps=n_iters,
                    **kwargs
                )
                p2_results[v2] = {"losses": losses, "path": path}
            p1_results[v1] = p2_results
        results[name] = p1_results

    return results

def plot(results: dict, config: dict, title="Hyperparameter Comparison", save_path=None, ax=None):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_algs = len(results)
    
    if ax is None:
        fig, axes = plt.subplots(1, n_algs, figsize=(9 * n_algs, 5), squeeze=False)
        axes_to_plot = axes[0]
        show = True
    else:
        axes_to_plot = [ax]
        show = False

    p1_name = config["param1_name"]
    p2_name = config["param2_name"]

    for ax_curr, (name, p1_dict) in zip(axes_to_plot, results.items()):
        for i, (v1, p2_dict) in enumerate(p1_dict.items()):
            color = colors[i % len(colors)]
            for j, (v2, data) in enumerate(p2_dict.items()):
                ls = _LINESTYLES[j % len(_LINESTYLES)]
                finite = [v for v in data['losses'] if np.isfinite(v)]
                final_str = f"{finite[-1]:.4g}" if finite else "diverged"
                label = f"{p1_name}={v1}, {p2_name}={v2} (Final: {final_str})"
                ax_curr.plot(data["losses"], label=label, linewidth=1.8, color=color, linestyle=ls)

        ax_curr.set_xlabel("Iteration", fontsize=12)
        ax_curr.set_ylabel("Loss", fontsize=12)
        ax_curr.set_title(name if show else title, fontsize=12, fontweight="bold")
        ax_curr.legend(title=f"{p1_name} · {p2_name}", framealpha=0.9, fontsize=8)
        ax_curr.grid(True, alpha=0.3)

    if show:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
