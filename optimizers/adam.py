import numpy as np


def adam(start_w, x, y,
         loss_func,
         gradient_func,
         lr=0.001,
         steps=100,
         beta1=0.9,
         beta2=0.999,
         eps=1e-8):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Parameters:
        start_w : initial weights (numpy array)
        x, y    : input data and targets
        loss_func : function computing loss (w, x, y)
        gradient_func : function computing gradient (w, x, y)
        lr      : learning rate (η)
        steps   : number of optimization steps
        beta1   : exponential decay rate for first moment
        beta2   : exponential decay rate for second moment
        eps     : numerical stability term

    Returns:
        final_w, loss_history, path_history
    """

    w = start_w.copy()

    m = np.zeros_like(w)  # first moment
    v = np.zeros_like(w)  # second moment

    loss_history = []
    path_history = [w.copy()]

    for t in range(1, steps + 1):
        # Divergence guard: pad with NaN so diverged runs end cleanly in plots
        if not np.all(np.isfinite(w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Compute loss
        current_loss = loss_func(w, x, y)
        if not np.isfinite(current_loss):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        loss_history.append(current_loss)

        # Gradient
        g = gradient_func(w, x, y)
        if not np.all(np.isfinite(g)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Parameter update
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        path_history.append(w.copy())

    return w, loss_history, np.array(path_history)