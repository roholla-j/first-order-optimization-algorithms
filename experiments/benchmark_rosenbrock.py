import numpy as np
from optimizers.gd       import gd
from optimizers.sgd      import sgd
from optimizers.momentum import momentum
from optimizers.nag      import nag
from optimizers.adam     import adam
from losses.rosenbrock   import rosenbrock, rosenbrock_gradient, distance_to_optimum
from utils.plotting      import plot_loss_curves, plot_rosenbrock_paths, plot_distance_to_optimum

OPTIMIZERS = {
    "GD":       (gd,       {"lr": 1e-3}),
    "SGD":      (sgd,      {"lr": 1e-3}),
    "Momentum": (momentum, {"lr": 1e-3, "beta": 0.9}),
    "NAG":      (nag,      {"lr": 1e-3, "beta": 0.9}),
    "Adam":     (adam,     {"lr": 1e-2}),
}

np.random.seed(0)
w0 = np.array([-1.0, 1.0])   # 2D starting point

loss_results = {}
path_results = {}
dist_results = {}

for name, (opt_fn, kwargs) in OPTIMIZERS.items():
    final_w, losses, path = opt_fn(
        start_w=w0,
        x=None, y=None,
        loss_func=rosenbrock,
        gradient_func=rosenbrock_gradient,
        steps=2000,
        **kwargs,
    )
    loss_results[name] = {"losses": losses}
    path_results[name] = path
    dist_results[name] = distance_to_optimum(path)
    print(f"{name:<14} final loss={losses[-1]:.6f}  "
          f"dist to optimum={dist_results[name][-1]:.6f}")

plot_loss_curves(loss_results,          title="Rosenbrock — loss curves")
plot_rosenbrock_paths(path_results,     title="Rosenbrock — optimizer trajectories")
plot_distance_to_optimum(dist_results,  title="Rosenbrock — distance to optimum")