import numpy as np
from losses.logistic import logistic_loss, logistic_gradient


class OptimizerRunner:
    """
    Adapts function-based optimizers (gd, sgd, momentum, nag, adam)
    to a uniform interface used by all metric modules.
    """

    def __init__(self, opt_fn, opt_kwargs: dict,
                 loss_fn=logistic_loss, grad_fn=logistic_gradient):
        """
        Parameters
        ----------
        opt_fn      : the optimizer function (e.g. gd, adam)
        opt_kwargs  : hyperparams to pass, e.g. {"lr": 0.1}
                      do NOT include start_w, x, y, loss_func, gradient_func
        loss_fn     : loss function
        grad_fn     : gradient function
        """
        self.opt_fn     = opt_fn
        self.opt_kwargs = opt_kwargs
        self.loss_fn    = loss_fn
        self.grad_fn    = grad_fn

    def run(self, X, y, w0: np.ndarray, n_iters: int) -> dict:
        final_w, losses, path = self.opt_fn(
            start_w=w0,
            x=X,
            y=y,
            loss_func=self.loss_fn,
            gradient_func=self.grad_fn,
            steps=n_iters,
            **self.opt_kwargs,
        )
        grad_norms = [
            float(np.linalg.norm(self.grad_fn(w, X, y))) for w in path[:-1]
        ]
        return {
            "w": final_w,
            "losses": losses,
            "final_loss": losses[-1],
            "path": path,
            "grad_norms": grad_norms,
        }


