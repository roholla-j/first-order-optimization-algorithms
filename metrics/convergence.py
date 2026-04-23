"""
Convergence metric
------------------
For each optimizer, tracks the full loss curve and reports:
  - losses            : list[float]  full trajectory
  - iters_to_eps      : dict[float, int|None]  first iteration below each loss epsilon
  - iters_to_grad_tol : dict[float, int|None]  first iteration where ‖∇f‖ < tol
  - auc               : float  area under loss curve (lower = faster overall)
  - wall_time         : float  seconds taken
  - final_loss        : float
  - final_grad_norm   : float
"""
import time
import numpy as np
from utils.runner import OptimizerRunner


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
    X, y       : training data
    config     : {
        n_iters   : int,
        epsilons  : list[float]   loss thresholds, e.g. [0.5, 0.35, 0.1]
        grad_tols : list[float]   gradient norm thresholds, e.g. [1e-2, 1e-3, 1e-4]
        w0_seed   : int,
    }
    """
    rng       = np.random.default_rng(config.get("w0_seed", 0))
    w0        = rng.standard_normal(X.shape[1]) * 0.01
    epsilons  = config.get("epsilons", [0.5, 0.35, 0.01])
    grad_tols = config.get("grad_tols", [])
    results   = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        runner = OptimizerRunner(opt_fn, opt_kwargs)

        t0  = time.perf_counter()
        out = runner.run(X, y, w0, config["n_iters"])
        t1  = time.perf_counter()

        losses     = out["losses"]
        grad_norms = out["grad_norms"]

        if len(losses) == 0 or np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            results[name] = {
                "losses":            [],
                "grad_norms":        [],
                "iters_to_eps":      {eps: None for eps in epsilons},
                "iters_to_grad_tol": {tol: None for tol in grad_tols},
                "auc":               None,
                "wall_time":         t1 - t0,
                "final_loss":        None,
                "final_grad_norm":   None,
            }
            continue

        # First iteration below each loss epsilon
        iters_to_eps = {}
        for eps in epsilons:
            iters_to_eps[eps] = next(
                (i for i, l in enumerate(losses) if l < eps), None
            )

        # First iteration where ‖∇f(w)‖ < tol
        iters_to_grad_tol = {}
        for tol in grad_tols:
            iters_to_grad_tol[tol] = next(
                (i for i, g in enumerate(grad_norms) if g < tol), None
            )

        results[name] = {
            "losses":            losses,
            "grad_norms":        grad_norms,
            "iters_to_eps":      iters_to_eps,
            "iters_to_grad_tol": iters_to_grad_tol,
            "auc":               float(np.trapezoid(losses) / len(losses)),
            "wall_time":         t1 - t0,
            "final_loss":        out["final_loss"],
            "final_grad_norm":   grad_norms[-1] if grad_norms else None,
        }

    return results


def summarize(results: dict, epsilons: list, grad_tols: list = None) -> None:
    grad_tols = grad_tols or []
    eps_headers  = "".join(f"  iters(ε={e})"    for e in epsilons)
    grad_headers = "".join(f"  iters(‖∇f‖<{t})" for t in grad_tols)

    print(f"\n{'Optimizer':<14} {'Final loss':>10} {'‖∇f‖ final':>12} "
          f"{'AUC':>8} {'Time(s)':>8}{eps_headers}{grad_headers}")
    print("-" * (54 + 14 * len(epsilons) + 16 * len(grad_tols)))

    for name, d in results.items():
        eps_cols  = "".join(f"  {str(d['iters_to_eps'][e]):>12}"      for e in epsilons)
        grad_cols = "".join(f"  {str(d['iters_to_grad_tol'][t]):>14}" for t in grad_tols)

        final_loss      = f"{d['final_loss']:.4f}"      if d['final_loss']      is not None else "None"
        final_grad_norm = f"{d['final_grad_norm']:.2e}" if d['final_grad_norm'] is not None else "None"
        auc             = f"{d['auc']:.2f}"             if d['auc']             is not None else "None"

        print(f"{name:<14} {final_loss:>10} {final_grad_norm:>12} "
              f"{auc:>8} {d['wall_time']:>8.4f}{eps_cols}{grad_cols}")
