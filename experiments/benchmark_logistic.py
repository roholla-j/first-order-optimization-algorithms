import numpy as np
from utils.data import load_data, train_test_split
from utils.plotting import (
    plot_robustness_dashboard,
    plot_sensitivity_heatmap,
    plot_sensitivity_line,
)

from optimizers.gd       import gd
from optimizers.sgd      import sgd
from optimizers.momentum import momentum
from optimizers.nag      import nag
from optimizers.adam     import adam

from metrics import robustness, sensitivity, lr_comparison, sgd_comparison, momentum_comparison

X, y            = load_data("data/australian_scale.txt")
X_tr, y_tr, *_ = train_test_split(X, y)

OPTIMIZERS = {
    "GD":       (gd,       {"lr": 0.1}),
    "SGD":      (sgd,      {"lr": 0.1}),
    "Momentum": (momentum, {"lr": 0.05, "beta": 0.9}),
    "NAG":      (nag,      {"lr": 0.05, "beta": 0.9}),
    "Adam":     (adam,     {"lr": 0.01}),
}

# ── 1. LEARNING RATE COMPARISON — full-batch GD ───────────────────────────────
# lr_cfg = {
#     "learning_rates": [2.0, 2.0, 2.5, 3.0],
#     "n_iters":        1000,
#     "w0_seed":        0,
# }
# lr_results = lr_comparison.run({"GD": (momentum, {"beta": 0.95})}, X_tr, y_tr, lr_cfg)
# lr_comparison.plot(
#     lr_results,
#     title="Full-Batch GD — Effect of Learning Rate\nLogistic Regression on Australian Credit Dataset",
#     save_path="experiments/lr_comparison.png",
# )

# ── 2. SGD — lr × batch-size COMPARISON ──────────────────────────────────────
sgd_cfg = {
    "learning_rates": [0.001, 0.1],
    "batch_sizes":    [1, 64],
    "n_epochs":       200,
    "w0_seed":        0,
}
sgd_results = sgd_comparison.run({"SGD": (sgd, {})}, X_tr, y_tr, sgd_cfg)
sgd_comparison.plot(
    sgd_results,
    title="SGD — lr × batch-size Comparison\nLogistic Regression on Australian Credit Dataset",
    save_path="experiments/sgd_comparison.png",
)

# # ── 2. ROBUSTNESS ─────────────────────────────────────────────────────────────
# rob_cfg = {
#     "n_iters":      1000,
#     "n_seeds":      20,
#     "w0_seed":      0,
#     "noise_levels": [0, 0.05, 0.1, 0.3],
# }
# rob_results = robustness.run(OPTIMIZERS, X_tr, y_tr, rob_cfg)
# robustness.summarize(rob_results)
# plot_robustness_dashboard(rob_results, title="Robustness — Logistic Regression ")

# # ── 3. SENSITIVITY — lr × beta  (Momentum, NAG) ───────────────────────────────
# sens_2d_cfg = {
#     "n_iters":   1000,
#     "w0_seed":   0,
#     "param_grid": {
#         "lr":   [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
#         "beta": [0.5, 0.7, 0.9, 0.95, 0.99],
#     },
# }
# sens_2d = sensitivity.run(
#     {"Momentum": OPTIMIZERS["Momentum"], "NAG": OPTIMIZERS["NAG"]},
#     X_tr, y_tr, sens_2d_cfg,
# )
# sensitivity.summarize(sens_2d)
# for name, d in sens_2d.items():
#     plot_sensitivity_heatmap(
#         d["grid"], d["row_vals"], d["col_vals"],
#         row_label=d["row_label"], col_label=d["col_label"],
#         title=f"Hyperparameter Sensitivity — {name}  (lr × beta)",
#     )

# # ── 4. SENSITIVITY — lr only  (GD, SGD, Adam) ────────────────────────────────
# sens_1d_cfg = {
#     "n_iters":   1000,
#     "w0_seed":   0,
#     "param_grid": {
#         "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
#     },
# }
# sens_1d = sensitivity.run(
#     {"GD": OPTIMIZERS["GD"], "SGD": OPTIMIZERS["SGD"], "Adam": OPTIMIZERS["Adam"]},
#     X_tr, y_tr, sens_1d_cfg,
# )
# sensitivity.summarize(sens_1d)
# plot_sensitivity_line(
#     sens_1d, param_name="lr",
#     title="Learning Rate Sensitivity — GD, SGD, Adam",
# )

# ── 3. MOMENTUM/NAG — lr × beta COMPARISON ───────────────────────────────────
# mom_cfg = {
#     "learning_rates": [3.7, 3.8, 2.0],
#     "betas":          [0.5, 0.9, 0.99],
#     "n_iters":        1000,
#     "w0_seed":        0,
# }
# mom_results = momentum_comparison.run(
#     {"Momentum": (momentum, {}), "NAG": (nag, {})},
#     X_tr, y_tr, mom_cfg,
# )
# momentum_comparison.plot(
#     mom_results,
#     title="Momentum & NAG — lr × beta Comparison\nLogistic Regression on Australian Credit Dataset",
#     save_path="experiments/momentum_comparison.png",
# )
