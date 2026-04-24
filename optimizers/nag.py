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
        gamma   : momentum coefficient (typically 0.9)

    Returns:
        final_w, loss_history, path_history
    """

    w = start_w.copy()
    v = np.zeros_like(w)

    loss_history = []
    path_history = [w.copy()]

    for _ in range(steps):
        # Divergence guard: pad with NaN so diverged runs end cleanly in plots
        if not np.all(np.isfinite(w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Record loss at current position
        current_loss = loss_func(w, x, y)
        if not np.isfinite(current_loss):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        loss_history.append(current_loss)

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

        # Update velocity (v_t = beta * v_{t-1} + lr * grad_at_lookahead)
        v = beta * v + lr * grad

        # Update parameters
        w = w - v

        # Save path
        path_history.append(w.copy())

    return w, loss_history, np.array(path_history)