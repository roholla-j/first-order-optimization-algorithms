import numpy as np

def nag(start_w, x, y,
        loss_func,
        gradient_func,
        lr=0.01,
        steps=20,
        beta=0.9):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    Parameters:
        start_w : initial weights (numpy array)
        x, y    : input data and targets
        loss_func : function that computes the loss (w, x, y)
        gradient_func : function that computes the gradient (w, x, y)
        lr      : learning rate (η)
        steps   : number of optimization steps
        beta    : momentum coefficient (typically 0.9)

    Returns:
        final_w, loss_history, path_history
    """
    w = start_w.copy()
    v = np.zeros_like(w)

    loss_history = []
    path_history = [w.copy()]

    for _ in range(steps):
        if not np.all(np.isfinite(w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Lookahead position: w_t - beta * v_{t-1}
        lookahead_w = w - beta * v
        if not np.all(np.isfinite(lookahead_w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Gradient evaluated at lookahead position
        grad = gradient_func(lookahead_w, x, y)
        if not np.all(np.isfinite(grad)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Update velocity and parameters
        v = beta * v + lr * grad
        w = w - v

        path_history.append(w.copy())

        current_loss = loss_func(w, x, y)
        if not np.isfinite(current_loss):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        loss_history.append(current_loss)

    return w, loss_history, np.array(path_history)
