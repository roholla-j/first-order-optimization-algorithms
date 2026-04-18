import numpy as np
import matplotlib.pyplot as plt

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


# ── Convergence ───────────────────────────────────────────────────────────────

def plot_loss_curves(results: dict, title="Loss curves", log_scale=True, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, data) in enumerate(results.items()):
        ax.plot(data["losses"], label=name, linewidth=1.8, color=COLORS[i % len(COLORS)])
    if log_scale:
        ax.set_yscale("log")
    _setup_ax(ax, title, "Iteration", "Loss")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_convergence_dashboard(results: dict, title="Convergence", save_path=None):
    """Loss curves (log scale) + AUC bar chart side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, (name, data) in enumerate(results.items()):
        losses = data["losses"]
        if losses:
            ax.semilogy(losses, label=name, linewidth=1.8, color=COLORS[i % len(COLORS)])
    _setup_ax(ax, "Loss curves (log scale)", "Iteration", "Loss")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    names = [n for n, d in results.items() if d.get("auc") is not None]
    aucs  = [results[n]["auc"] for n in names]
    bars  = ax.bar(names, aucs,
                   color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    _setup_ax(ax, "Area Under Loss Curve\n(lower = faster convergence)", "Optimizer", "AUC")
    ax.tick_params(axis="x", rotation=15)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_convergence_separate(results: dict, save_path=None):
    """One subplot per optimizer (log scale)."""
    names = list(results.keys())
    n     = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.semilogy(results[name]["losses"], color=COLORS[i % len(COLORS)], linewidth=1.8)
        _setup_ax(ax, name, "", "Loss")
    axes[-1].set_xlabel("Iteration")
    plt.suptitle("Convergence — per optimizer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


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
                              title="Sensitivity", save_path=None):
    """
    2D heatmap with explicit tick labels — handles non-linear (log) spacing correctly.
    Annotates each cell when the grid is small enough to read.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    display_grid = grid.copy()
    finite = display_grid[np.isfinite(display_grid)]
    if len(finite) > 0:
        vmin = float(finite.min())
        vmax = float(np.percentile(finite, 95))  # clip outliers for colour contrast
        display_grid = np.clip(display_grid, vmin, vmax)
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(display_grid, aspect="auto", origin="lower", cmap="viridis_r",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Final loss  (lower = better)")

    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([f"{v:.3g}" for v in col_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([f"{v:.3g}" for v in row_vals])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title, fontsize=11, fontweight="bold")

    if grid.size <= 60 and len(finite) > 0:
        mid = (vmin + vmax) / 2
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = grid[i, j]
                if np.isfinite(val):
                    txt_color = "white" if display_grid[i, j] > mid else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=txt_color)

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

def plot_rosenbrock_paths(paths: dict, title="Rosenbrock trajectories", save_path=None):
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contour(X1, X2, np.log1p(Z), levels=30, cmap="gray", alpha=0.5)

    for i, (name, path) in enumerate(paths.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(path[:, 0], path[:, 1], marker=".", markersize=3,
                label=name, color=color, alpha=0.8)
        ax.plot(*path[0],  "o", markersize=7,  color=color, markeredgecolor="black")
        ax.plot(*path[-1], "*", markersize=11, color=color, markeredgecolor="black")

    ax.plot(1, 1, "r*", markersize=15, label="optimum (1,1)", zorder=5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.9)
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
