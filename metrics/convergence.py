"""
Convergence metric
------------------
For each optimizer, tracks the full loss curve and reports:
  - losses       : list[float]  full trajectory
  - iters_to_eps : dict[float, int|None]  first iteration below each epsilon
  - auc          : float  area under loss curve (lower = faster overall)
  - wall_time    : float  seconds taken
  - final_loss   : float
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
        n_iters  : int,
        epsilons : list[float]   e.g. [0.5, 0.35, 0.1]
        w0_seed  : int,
    }
    """
    rng      = np.random.default_rng(config.get("w0_seed", 0))
    w0       = rng.standard_normal(X.shape[1]) * 0.01
    epsilons = config.get("epsilons", [0.5, 0.35, 0.01])
    results  = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        runner = OptimizerRunner(opt_fn, opt_kwargs)

        t0  = time.perf_counter()
        out = runner.run(X, y, w0, config["n_iters"])
        t1  = time.perf_counter()

        losses = out["losses"]
        if len(losses) == 0 or np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            results[name] = {
                "losses": [],
                "iters_to_eps": {eps: None for eps in epsilons},
                "auc": None,
                "wall_time": t1 - t0,
                "final_loss": None,
            }
            continue

        # First iteration below each epsilon threshold
        iters_to_eps = {}
        for eps in epsilons:
            step = None
            for i, l in enumerate(losses):
                if l < eps:
                    step = i
                    break
            iters_to_eps[eps] = step

        results[name] = {
            "losses":       losses,
            "iters_to_eps": iters_to_eps,
            "auc": float(np.trapezoid(losses) / len(losses)),
            "wall_time":    t1 - t0,
            "final_loss":   out["final_loss"],
        }

    return results


def summarize(results: dict, epsilons: list) -> None:
    eps_headers = "".join(f"  iters(ε={e})" for e in epsilons)
    print(f"\n{'Optimizer':<14} {'Final loss':>10} {'AUC':>10} {'Time(s)':>8}{eps_headers}")
    print("-" * (44 + 14 * len(epsilons)))
    for name, d in results.items():
        eps_cols = "".join(
            f"  {str(d['iters_to_eps'][e]):>12}" for e in epsilons
        )
        final_loss = f"{d['final_loss']:.4f}" if d['final_loss'] is not None else "None"
        auc = f"{d['auc']:.2f}" if d['auc'] is not None else "None"

        print(f"{name:<14} {final_loss:>10} {auc:>10} "
              f"{d['wall_time']:>8.4f}{eps_cols}")