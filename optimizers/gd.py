
import numpy as np
def gd(start_w, x, y, 
                     loss_func, 
                     gradient_func, 
                     lr=0.01, 
                     steps=20):
    """
    Generic Gradient Descent optimizer.
    
    Parameters:
        start_w : initial weights (numpy array)
        x, y    : input data and targets
        loss_func : function that computes the loss (signature: loss_func(w, x, y))
        gradient_func : function that computes the gradient (signature: gradient_func(w, x, y))
        lr      : learning rate
        steps   : number of optimization steps
    
    Returns:
        final_w, loss_history, path_history
    """
    w = start_w.copy()
    loss_history = []
    path_history = [w.copy()]

    for _ in range(steps):
        # Divergence guard: if w blew up, stop and pad with NaN
        # (NaN values are skipped by matplotlib, so the curve just ends cleanly)
        if not np.all(np.isfinite(w)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break

        # Compute and record loss
        current_loss = loss_func(w, x, y)
        if not np.isfinite(current_loss):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        loss_history.append(current_loss)

        # Compute gradient and update weights
        grad = gradient_func(w, x, y)
        if not np.all(np.isfinite(grad)):
            loss_history.extend([np.nan] * (steps - len(loss_history)))
            break
        w = w - lr * grad

        # Save path for visualization
        path_history.append(w.copy())

    return w, loss_history, np.array(path_history)