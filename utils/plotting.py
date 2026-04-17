import numpy as np
import matplotlib.pyplot as plt


def _setup_plot(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_loss_curves(results: dict, title="Loss curves", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        ax.plot(data["losses"], label=name)

    _setup_plot(ax, title, "Iteration", "Loss")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_robustness_boxplot(results: dict, title="Init robustness", save_path=None):
    """Box plot of final loss variance across init seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names       = list(results.keys())
    final_losses = [results[n]["init"]["final_losses"] for n in names]
    ax.boxplot(final_losses, labels=names, patch_artist=True)
    ax.set_ylabel("Final loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_noise_robustness(results: dict, title="Noise robustness", save_path=None):
    """Line plot: final loss vs noise level for each optimizer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, d in results.items():
        noise_levels = list(d["noise"].keys())
        losses = [d["noise"][s] for s in noise_levels]
        ax.plot(noise_levels, losses, marker="o", label=name)

    _setup_plot(ax, title, "Noise std", "Final loss")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sensitivity_heatmap(grid, row_vals, col_vals,
                              row_label="", col_label="",
                              title="Sensitivity", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[col_vals[0], col_vals[-1], row_vals[0], row_vals[-1]])
    plt.colorbar(im, ax=ax, label="Final loss")
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rosenbrock_paths(paths: dict, title="Rosenbrock trajectories", save_path=None):
    """
    Overlay optimizer paths on the Rosenbrock contour (2D only).

    Parameters
    ----------
    paths : {name: np.ndarray of shape (n_iters+1, 2)}
    """
    # Build contour grid
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(X1, X2, np.log1p(Z), levels=30, cmap="gray", alpha=0.5)

    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], marker=".", markersize=3, label=name)
        ax.plot(*path[0],  "o", markersize=6, color="black")   # start
        ax.plot(*path[-1], "*", markersize=10)                  # end

    ax.plot(1, 1, "r*", markersize=14, label="optimum (1,1)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_distance_to_optimum(distances: dict, title="Distance to optimum", save_path=None):
    """
    Plot ‖w_t - w*‖ over iterations for each optimizer.

    Parameters
    ----------
    distances : {name: list[float]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, dists in distances.items():
        ax.plot(dists, label=name)

    _setup_plot(ax, title, "Iteration", "‖w − w*‖")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
    
def plot_convergence_separate(results: dict, save_path=None):
    """
    One subplot per optimizer showing its loss curve over all iterations.
    """
    names = list(results.keys())
    n     = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        ax.plot(results[name]["losses"])
        _setup_plot(ax, name, "", "Loss")

    axes[-1].set_xlabel("Iteration")
    plt.suptitle("Convergence speed — per optimizer", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()