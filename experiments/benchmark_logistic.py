import numpy as np
from utils.data import load_data, train_test_split
from utils.plotting import (
    plot_convergence_dashboard,
    plot_robustness_dashboard,
    plot_sensitivity_heatmap,
    plot_sensitivity_line,
)

from optimizers.gd       import gd
from optimizers.sgd      import sgd
from optimizers.momentum import momentum
from optimizers.nag      import nag
from optimizers.adam     import adam

from metrics import convergence, robustness, sensitivity

X, y            = load_data("data/australian_scale.txt")
X_tr, y_tr, *_ = train_test_split(X, y)

OPTIMIZERS = {
    "GD":       (gd,       {"lr": 0.1}),
    "SGD":      (sgd,      {"lr": 0.1}),
    "Momentum": (momentum, {"lr": 0.05, "beta": 0.9}),
    "NAG":      (nag,      {"lr": 0.05, "beta": 0.9}),
    "Adam":     (adam,     {"lr": 0.01}),
}

# ── 1. CONVERGENCE ────────────────────────────────────────────────────────────
conv_cfg = {
    "n_iters":  1000,
    "epsilons": [0.5, 0.35, 0.2],
    "w0_seed":  0,
}
conv_results = convergence.run(OPTIMIZERS, X_tr, y_tr, conv_cfg)
convergence.summarize(conv_results, epsilons=conv_cfg["epsilons"])
plot_convergence_dashboard(conv_results, title="Convergence — Logistic Regression (RCV1)")

# ── 2. ROBUSTNESS ─────────────────────────────────────────────────────────────
rob_cfg = {
    "n_iters":      1000,
    "n_seeds":      20,
    "w0_seed":      0,
    "noise_levels": [0, 0.05, 0.1, 0.3],
}
rob_results = robustness.run(OPTIMIZERS, X_tr, y_tr, rob_cfg)
robustness.summarize(rob_results)
plot_robustness_dashboard(rob_results, title="Robustness — Logistic Regression (RCV1)")

# ── 3. SENSITIVITY — lr × beta  (Momentum, NAG) ───────────────────────────────
sens_2d_cfg = {
    "n_iters":   1000,
    "w0_seed":   0,
    "param_grid": {
        "lr":   [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "beta": [0.5, 0.7, 0.9, 0.95, 0.99],
    },
}
sens_2d = sensitivity.run(
    {"Momentum": OPTIMIZERS["Momentum"], "NAG": OPTIMIZERS["NAG"]},
    X_tr, y_tr, sens_2d_cfg,
)
sensitivity.summarize(sens_2d)
for name, d in sens_2d.items():
    plot_sensitivity_heatmap(
        d["grid"], d["row_vals"], d["col_vals"],
        row_label=d["row_label"], col_label=d["col_label"],
        title=f"Hyperparameter Sensitivity — {name}  (lr × beta)",
    )

# ── 4. SENSITIVITY — lr only  (GD, SGD, Adam) ────────────────────────────────
sens_1d_cfg = {
    "n_iters":   1000,
    "w0_seed":   0,
    "param_grid": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    },
}
sens_1d = sensitivity.run(
    {"GD": OPTIMIZERS["GD"], "SGD": OPTIMIZERS["SGD"], "Adam": OPTIMIZERS["Adam"]},
    X_tr, y_tr, sens_1d_cfg,
)
sensitivity.summarize(sens_1d)
plot_sensitivity_line(
    sens_1d, param_name="lr",
    title="Learning Rate Sensitivity — GD, SGD, Adam",
)
