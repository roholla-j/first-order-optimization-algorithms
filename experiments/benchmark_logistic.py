

import numpy as np
from utils.data import load_data, train_test_split
from utils.plotting import (plot_loss_curves, plot_robustness_boxplot,
                             plot_noise_robustness, plot_sensitivity_heatmap, plot_convergence_separate)

from optimizers.gd       import gd
from optimizers.sgd      import sgd
from optimizers.momentum import momentum
from optimizers.nag      import nag
from optimizers.adam     import adam

from metrics import convergence, robustness, sensitivity

X, y             = load_data("data/australian_scale.txt")
X_tr, y_tr, *_  = train_test_split(X, y)

OPTIMIZERS = {
    "GD":       (gd,       {"lr": 0.1}),
    "SGD":      (sgd,      {"lr": 0.05}),
    "Momentum": (momentum, {"lr": 0.05, "beta": 0.9}),
    "NAG":      (nag,      {"lr": 0.05, "beta": 0.9}),
    "Adam":     (adam,     {"lr": 0.01}),
}

# ── 1. CONVERGENCE ───────────────────────────────────────────────────────────
conv_cfg = {
    "n_iters":  500,
    "epsilons": [0.5, 0.35, 0.1],
    "w0_seed":  0,
}
conv_results = convergence.run(OPTIMIZERS, X_tr, y_tr, conv_cfg)
convergence.summarize(conv_results, epsilons=conv_cfg["epsilons"])
plot_loss_curves(conv_results, title="Convergence — loss curves")

plot_loss_curves(conv_results, title="Convergence — all optimizers")
plot_convergence_separate(conv_results)                    # per optimizer

# ── 2. ROBUSTNESS ────────────────────────────────────────────────────────────
rob_cfg = {
    "n_iters":      300,
    "n_seeds":      20,
    "w0_seed":      0,
    "noise_levels": [0, 0.05, 0.1, 0.2],
}
rob_results = robustness.run(OPTIMIZERS, X_tr, y_tr, rob_cfg)
robustness.summarize(rob_results)
plot_robustness_boxplot(rob_results, title="Init robustness — final loss distribution")
plot_noise_robustness(rob_results,   title="Noise robustness — degradation by noise level")

# ── 3. SENSITIVITY ───────────────────────────────────────────────────────────
sens_cfg = {
    "n_iters":   300,
    "w0_seed":   0,
    "param_grid": {
        "lr":   [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "beta": [0.5, 0.7, 0.9, 0.95, 0.99],
    },
}
# Run sensitivity for optimizers that have a beta parameter
sens_results = sensitivity.run(
    {"Momentum": OPTIMIZERS["Momentum"], "NAG": OPTIMIZERS["NAG"]},
    X_tr, y_tr, sens_cfg
)
sensitivity.summarize(sens_results)
for name, d in sens_results.items():
    plot_sensitivity_heatmap(
        d["grid"], d["row_vals"], d["col_vals"],
        row_label=d["row_label"], col_label=d["col_label"],
        title=f"Sensitivity — {name}",
    )