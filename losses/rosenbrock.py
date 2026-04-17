import numpy as np


def rosenbrock(w, *args):
    """
    Rosenbrock function for n-dimensional input.

    f(x) = sum_{i=1}^{n-1} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    Global minimum: f(1, 1, ..., 1) = 0

    Parameters
    ----------
    w    : np.ndarray of shape (n,)
    *args: ignored — present so the signature matches logistic_loss(w, X, y)
           and can be passed to OptimizerRunner without modification

    Returns
    -------
    float : scalar loss value
    """
    return float(np.sum(
        100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1 - w[:-1]) ** 2
    ))


def rosenbrock_gradient(w, *args):
    """
    Analytical gradient of the Rosenbrock function.

    d f / d x_i =
      -400 * x_i * (x_{i+1} - x_i^2) - 2*(1 - x_i)   for i = 1..n-1
      +200 * (x_i - x_{i-1}^2)                          for i = 2..n

    Parameters
    ----------
    w    : np.ndarray of shape (n,)
    *args: ignored — matches gradient_func(w, X, y) signature

    Returns
    -------
    np.ndarray of shape (n,) : gradient vector
    """
    n    = len(w)
    grad = np.zeros(n)

    # Interior contribution from being x_i (the "left" variable)
    grad[:-1] += -400.0 * w[:-1] * (w[1:] - w[:-1] ** 2) - 2.0 * (1.0 - w[:-1])

    # Interior contribution from being x_{i+1} (the "right" variable)
    grad[1:]  += 200.0 * (w[1:] - w[:-1] ** 2)

    return grad


def distance_to_optimum(path_history: np.ndarray) -> list:
    """
    Euclidean distance from each point in the path to the global minimum w* = (1,...,1).

    Parameters
    ----------
    path_history : np.ndarray of shape (n_iters+1, n_dims)
                   as returned by your optimizers

    Returns
    -------
    list of floats, one per step
    """
    w_star = np.ones(path_history.shape[1])
    return [float(np.linalg.norm(w - w_star)) for w in path_history]