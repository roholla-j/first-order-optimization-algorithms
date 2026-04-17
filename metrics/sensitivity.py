"""
Hyperparameter sensitivity metric
----------------------------------
Sweeps one or two hyperparameters for each optimizer and reports:
  - grid              : 2-D array of final losses
  - best_cfg          : dict of best hyperparameter values found
  - best_loss         : float
  - sensitivity_score : coefficient of variation across grid
                        (std / mean) — higher means more sensitive
"""
import numpy as np
from utils.runner import OptimizerRunner


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
    X, y       : training data
    config     : {
        n_iters    : int,
        w0_seed    : int,
        param_grid : {"lr": [...], "beta": [...]}  # second param optional
    }
    """
    param_grid = config["param_grid"]
    params     = list(param_grid.keys())
    p1_name    = params[0]
    p2_name    = params[1] if len(params) > 1 else None
    p1_vals    = param_grid[p1_name]
    p2_vals    = param_grid[p2_name] if p2_name else [None]

    rng     = np.random.default_rng(config.get("w0_seed", 0))
    w0      = rng.standard_normal(X.shape[1]) * 0.01
    results = {}

    for name, (opt_fn, base_kwargs) in optimizers.items():
        grid      = np.full((len(p1_vals), len(p2_vals)), np.nan)
        best_loss = np.inf
        best_cfg  = {}

        for i, v1 in enumerate(p1_vals):
            for j, v2 in enumerate(p2_vals):
                kwargs_copy = {**base_kwargs, p1_name: v1}
                if p2_name and v2 is not None:
                    kwargs_copy[p2_name] = v2

                runner     = OptimizerRunner(opt_fn, kwargs_copy)
                out        = runner.run(X, y, w0, config["n_iters"])
                loss = out["final_loss"]

                if np.isnan(loss) or np.isinf(loss):
                    grid[i, j] = np.nan
                    continue

                grid[i, j] = loss

                if loss < best_loss:
                    best_loss = loss
                    best_cfg  = {p1_name: v1, **(
                        {p2_name: v2} if p2_name else {}
                    )}

        if best_loss == np.inf:
            best_loss = None
            best_cfg = {}

        flat = grid.flatten()
        flat = flat[~np.isnan(flat)]

        if len(flat) == 0:
            score = None
        else:
            score = float(flat.std() / flat.mean()) if flat.mean() != 0 else 0.0

        results[name] = {
            "grid":              grid,
            "row_vals":          p1_vals,
            "col_vals":          p2_vals if p2_name else [0],
            "row_label":         p1_name,
            "col_label":         p2_name or "",
            "best_cfg":          best_cfg,
            "best_loss":         best_loss,
            "sensitivity_score": score,
        }

    return results


def summarize(results: dict) -> None:
    print(f"\n{'Optimizer':<14} {'Best loss':>10} {'Sensitivity':>13}  Best config")
    print("-" * 65)
    for name, d in results.items():
        cfg_str = ", ".join(f"{k}={v}" for k, v in d["best_cfg"].items())
        best_loss = f"{d['best_loss']:.4f}" if d["best_loss"] is not None else "None"
        score     = f"{d['sensitivity_score']:.4f}" if d["sensitivity_score"] is not None else "None"

        print(f"{name:<14} {best_loss:>10} "
              f"{score:>13}  {cfg_str}")