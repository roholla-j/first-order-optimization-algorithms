

"""
Robustness metric
-----------------
Two separate sub-experiments:

1. init_robustness  — vary w0 seed, fix data
   reports variance in final loss due to initialisation alone

2. noise_robustness — fix w0, vary noise level on X
   reports how gracefully each optimizer degrades as data gets noisier
"""
import numpy as np
from utils.runner import OptimizerRunner


def _run_seeds(opt_fn, opt_kwargs, X, y, w0_base_seed, n_seeds, n_iters):
    """Vary w0 seed, keep data fixed."""
    final_losses = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + w0_base_seed)
        w0  = rng.standard_normal(X.shape[1]) * 0.01
        out = OptimizerRunner(opt_fn, opt_kwargs).run(X, y, w0, n_iters)
        loss = out["final_loss"]
        if np.isnan(loss) or np.isinf(loss):
            continue
        final_losses.append(loss)
    return final_losses


def _run_noise(opt_fn, opt_kwargs, X, y, w0_seed, noise_levels, n_iters):
    """Fix w0, sweep noise levels."""
    rng = np.random.default_rng(w0_seed)
    w0  = rng.standard_normal(X.shape[1]) * 0.01
    noise_rng = np.random.default_rng(0)

    results = {}
    for std in noise_levels:
        X_noisy = X + noise_rng.normal(0, std, X.shape) if std > 0 else X
        out = OptimizerRunner(opt_fn, opt_kwargs).run(X_noisy, y, w0, n_iters)
        loss = out["final_loss"]
        results[std] = None if (np.isnan(loss) or np.isinf(loss)) else loss
    return results


def run(optimizers: dict, X, y, config: dict) -> dict:
    """
    Parameters
    ----------
    optimizers : {name: (optimizer_fn, kwargs)}
    X, y       : training data
    config     : {
        n_iters      : int,
        n_seeds      : int,           # for init robustness  (e.g. 20)
        w0_seed      : int,           # base seed            (e.g. 0)
        noise_levels : list[float],   # for noise robustness (e.g. [0, 0.05, 0.1, 0.2])
    }

    Returns
    -------
    {name: {
        "init":  {"final_losses": [...], "mean": float, "std": float, "worst": float},
        "noise": {noise_std: final_loss, ...}
    }}
    """
    n_iters      = config["n_iters"]
    n_seeds      = config.get("n_seeds", 20)
    w0_seed      = config.get("w0_seed", 0)
    noise_levels = config.get("noise_levels", [0, 0.05, 0.1, 0.2])
    results      = {}

    for name, (opt_fn, opt_kwargs) in optimizers.items():
        # 1. Init robustness
        final_losses = _run_seeds(opt_fn, opt_kwargs, X, y, w0_seed, n_seeds, n_iters)

        if len(final_losses) == 0:
            results[name] = {
                "init": {
                    "final_losses": [],
                    "mean": None,
                    "std": None,
                    "worst": None,
                    "success_rate": 0.0,
                },
                "noise": {},
            }
            continue

        arr = np.array(final_losses)
        success_rate = len(final_losses) / n_seeds

        # 2. Noise robustness
        noise_results = _run_noise(opt_fn, opt_kwargs, X, y, w0_seed,
                                   noise_levels, n_iters)

        baseline = noise_results.get(0)
        degradation = {
            std: (loss - baseline) if (loss is not None and baseline is not None) else None
            for std, loss in noise_results.items()
        }

        results[name] = {
            "init": {
                "final_losses": final_losses,
                "mean":  float(arr.mean()),
                "std":   float(arr.std()),
                "worst": float(arr.max()),
                "success_rate": success_rate,
            },
            "noise": noise_results,
            "degradation": degradation,
        }

    return results


def summarize(results: dict) -> None:
    # Init robustness table
    print(f"\n--- Init robustness ({list(results.values())[0]['init']['mean'].__class__.__name__}) ---")
    print(f"\n{'Optimizer':<14} {'Mean':>10} {'Std':>8} {'Worst':>10}")
    print("-" * 44)
    for name, d in results.items():
        i = d["init"]
        mean  = f"{i['mean']:.4f}"  if i['mean']  is not None else "None"
        std   = f"{i['std']:.4f}"   if i['std']   is not None else "None"
        worst = f"{i['worst']:.4f}" if i['worst'] is not None else "None"

        print(f"{name:<14} {mean:>10} {std:>8} {worst:>10}")

    # Noise robustness table
    noise_levels = list(list(results.values())[0]["noise"].keys())
    headers      = "".join(f"  noise={s}" for s in noise_levels)
    print(f"\n--- Noise robustness ---")
    print(f"\n{'Optimizer':<14}{headers}")
    print("-" * (14 + 10 * len(noise_levels)))
    for name, d in results.items():
        row = "".join(f"  {d['noise'][s]:>8.4f}" for s in noise_levels)
        print(f"{name:<14}{row}")