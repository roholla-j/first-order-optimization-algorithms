import math
import numpy as np
import matplotlib.pyplot as plt

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def make_axes_grid(n_plots, w_per_plot=9, h_per_row=6, max_cols=2):
    """
    Create a figure with *n_plots* subplots arranged in a grid with at most
    *max_cols* columns.  Any unused grid cells are hidden.

    Returns
    -------
    fig, axes_flat : Figure and a flat list of *n_plots* Axes (ready to index)
    """
    ncols = min(n_plots, max_cols)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(w_per_plot * ncols, h_per_row * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    # Hide any extra cells
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)
    return fig, list(axes_flat[:n_plots])


# ── Robustness ────────────────────────────────────────────────────────────────

def plot_robustness_boxplot(results: dict, title="Init robustness", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    names        = list(results.keys())
    final_losses = [results[n]["init"]["final_losses"] for n in names]
    bp = ax.boxplot(final_losses, labels=names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Final loss")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_noise_robustness(results: dict, title="Noise robustness", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, d) in enumerate(results.items()):
        noise_levels = list(d["noise"].keys())
        losses       = [d["noise"][s] for s in noise_levels]
        valid = [(nl, l) for nl, l in zip(noise_levels, losses) if l is not None]
        if valid:
            nl_v, l_v = zip(*valid)
            ax.plot(nl_v, l_v, marker="o", label=name, linewidth=1.8,
                    color=COLORS[i % len(COLORS)])
    _setup_ax(ax, title, "Noise std", "Final loss")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_robustness_dashboard(results: dict, title="Robustness", save_path=None):
    """Init robustness boxplot + noise robustness line plot side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    names        = list(results.keys())
    final_losses = [results[n]["init"]["final_losses"] for n in names]
    bp = ax.boxplot(final_losses, labels=names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Final loss")
    ax.set_title("Init robustness\n(final loss over 20 random starts)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for i, (name, d) in enumerate(results.items()):
        noise_levels = list(d["noise"].keys())
        losses       = [d["noise"][s] for s in noise_levels]
        valid = [(nl, l) for nl, l in zip(noise_levels, losses) if l is not None]
        if valid:
            nl_v, l_v = zip(*valid)
            ax.plot(nl_v, l_v, marker="o", label=name, linewidth=1.8,
                    color=COLORS[i % len(COLORS)])
    _setup_ax(ax, "Noise robustness\n(final loss vs input noise level)",
              "Noise std", "Final loss")
    ax.legend(framealpha=0.9)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Sensitivity ───────────────────────────────────────────────────────────────

def plot_sensitivity_heatmap(grid, row_vals, col_vals,
                              row_label="lr", col_label="beta",
                              title="Sensitivity", save_path=None, ax=None):
    """
    2D heatmap with explicit tick labels — handles non-linear (log) spacing correctly.
    Annotates each cell when the grid is small enough to read.
    """
    if ax is None:
        fig, ax_to_plot = plt.subplots(figsize=(8, 6))
        show = True
    else:
        ax_to_plot = ax
        show = False

    display_grid = grid.copy()
    finite = display_grid[np.isfinite(display_grid)]
    if len(finite) > 0:
        vmin = float(finite.min())
        vmax = float(np.percentile(finite, 95))  # clip outliers for colour contrast
        display_grid = np.clip(display_grid, vmin, vmax)
    else:
        vmin, vmax = 0.0, 1.0

    im = ax_to_plot.imshow(display_grid, aspect="auto", origin="lower", cmap="viridis_r",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax_to_plot, label="Final loss  (lower = better)")

    ax_to_plot.set_xticks(range(len(col_vals)))
    ax_to_plot.set_xticklabels([f"{v:.3g}" for v in col_vals], rotation=45, ha="right")
    ax_to_plot.set_yticks(range(len(row_vals)))
    ax_to_plot.set_yticklabels([f"{v:.3g}" for v in row_vals])
    ax_to_plot.set_xlabel(col_label)
    ax_to_plot.set_ylabel(row_label)
    ax_to_plot.set_title(title, fontsize=11, fontweight="bold")

    if grid.size <= 60 and len(finite) > 0:
        mid = (vmin + vmax) / 2
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = grid[i, j]
                if np.isfinite(val):
                    txt_color = "white" if display_grid[i, j] > mid else "black"
                    ax_to_plot.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=txt_color)

    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def plot_sensitivity_line(results: dict, param_name="lr",
                          title="Learning rate sensitivity",
                          log_x=True, save_path=None):
    """Line plot of final loss vs a single hyperparameter for multiple optimizers."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, d) in enumerate(results.items()):
        param_vals = d["row_vals"]
        losses     = d["grid"][:, 0]
        valid = [(p, float(l)) for p, l in zip(param_vals, losses) if np.isfinite(l)]
        if valid:
            pv, lv = zip(*valid)
            ax.plot(pv, lv, marker="o", label=name, linewidth=1.8,
                    color=COLORS[i % len(COLORS)])
    if log_x:
        ax.set_xscale("log")
    _setup_ax(ax, title, param_name, "Final loss")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Rosenbrock ────────────────────────────────────────────────────────────────

def plot_rosenbrock_paths(paths: dict, title="Rosenbrock trajectories", save_path=None, ax=None):
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2

    if ax is None:
        fig, ax_to_plot = plt.subplots(figsize=(9, 7))
        show = True
    else:
        ax_to_plot = ax
        show = False

    ax_to_plot.contour(X1, X2, np.log1p(Z), levels=30, cmap="gray", alpha=0.5)

    for i, (name, path) in enumerate(paths.items()):
        color = COLORS[i % len(COLORS)]
        ax_to_plot.plot(path[:, 0], path[:, 1], marker=".", markersize=3,
                label=name, color=color, alpha=0.8)
        ax_to_plot.plot(*path[0],  "o", markersize=7,  color=color, markeredgecolor="black")
        ax_to_plot.plot(*path[-1], "*", markersize=11, color=color, markeredgecolor="black")

    ax_to_plot.plot(1, 1, "r*", markersize=15, label="optimum (1,1)", zorder=5)
    ax_to_plot.set_xlabel("x₁")
    ax_to_plot.set_ylabel("x₂")
    ax_to_plot.set_title(title, fontsize=11, fontweight="bold")
    ax_to_plot.legend(framealpha=0.9)
    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

def plot_quadratic_paths(paths: dict, A, b, title="Quadratic Bowl trajectories", save_path=None, ax=None):
    from losses.quadratic import quadratic_loss
    # Calculate the optimum
    w_star = np.linalg.inv(A) @ b

    # We want to make sure the meshgrid covers both the start points and the optimum
    all_points = np.vstack([p for p in paths.values()] + [w_star])
    x1_min, x1_max = all_points[:, 0].min(), all_points[:, 0].max()
    x2_min, x2_max = all_points[:, 1].min(), all_points[:, 1].max()

    # Add some padding
    padding_x1 = max(0.5, (x1_max - x1_min) * 0.2)
    padding_x2 = max(0.5, (x2_max - x2_min) * 0.2)

    x1 = np.linspace(x1_min - padding_x1, x1_max + padding_x1, 200)
    x2 = np.linspace(x2_min - padding_x2, x2_max + padding_x2, 200)
    X1, X2 = np.meshgrid(x1, x2)

    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Z = np.array([quadratic_loss(p, A, b) for p in pts]).reshape(X1.shape)

    if ax is None:
        fig, ax_to_plot = plt.subplots(figsize=(9, 7))
        show = True
    else:
        ax_to_plot = ax
        show = False

    ax_to_plot.contour(X1, X2, Z, levels=25, cmap="viridis", alpha=0.5)

    for i, (name, path) in enumerate(paths.items()):
        color = COLORS[i % len(COLORS)]
        ax_to_plot.plot(path[:, 0], path[:, 1], marker=".", markersize=3,
                label=name, color=color, alpha=0.8)
        ax_to_plot.plot(*path[0],  "o", markersize=7,  color=color, markeredgecolor="black")
        ax_to_plot.plot(*path[-1], "*", markersize=11, color=color, markeredgecolor="black")

    ax_to_plot.plot(*w_star, "r*", markersize=15, label="optimum (w*)", zorder=5)
    ax_to_plot.set_xlabel("w₁")
    ax_to_plot.set_ylabel("w₂")
    ax_to_plot.set_title(title, fontsize=11, fontweight="bold")
    ax_to_plot.legend(framealpha=0.9)
    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def plot_distance_to_optimum(distances: dict, title="Distance to optimum", save_path=None):
    _, ax = plt.subplots(figsize=(9, 5))
    for i, (name, dists) in enumerate(distances.items()):
        ax.semilogy(dists, label=name, linewidth=1.8, color=COLORS[i % len(COLORS)])
    _setup_ax(ax, title, "Iteration", "‖w − w*‖  (log scale)")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
