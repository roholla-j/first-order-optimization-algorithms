import numpy as np


def adam(start_w, x, y,
         loss_func,
         gradient_func,
         lr=0.001,
         steps=100,
         beta1=0.9,
         beta2=0.999,
         eps=1e-8,
         batch_size=None,
         seed=42):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Parameters:
        start_w    : initial weights (numpy array)
        x, y       : input data and targets
        loss_func  : function computing loss (w, x, y)
        gradient_func : function computing gradient (w, x, y)
        lr         : learning rate (η)
        steps      : number of optimization steps
        beta1      : exponential decay rate for first moment
        beta2      : exponential decay rate for second moment
        eps        : numerical stability term
        batch_size : mini-batch size; None means full-batch
        seed       : random seed for mini-batch sampling

    Returns:
        final_w, loss_history, path_history
    """
    w = start_w.copy()

    m = np.zeros_like(w)
    v = np.zeros_like(w)

    loss_history = []
    path_history = [w.copy()]

    n = x.shape[0] if hasattr(x, "shape") else 1
    if batch_size is not None:
        np.random.seed(seed)

    for t in range(1, steps + 1):
        if not np.all(np.isfinite(w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        if batch_size is not None and n > 1:
            indices = np.random.choice(n, min(batch_size, n), replace=False)
            x_batch, y_batch = x[indices], y[indices]
        else:
            x_batch, y_batch = x, y

        g = gradient_func(w, x_batch, y_batch)
        if not np.all(np.isfinite(g)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        path_history.append(w.copy())

        current_loss = loss_func(w, x, y)
        if not np.isfinite(current_loss):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        loss_history.append(current_loss)

    return w, loss_history, np.array(path_history)
