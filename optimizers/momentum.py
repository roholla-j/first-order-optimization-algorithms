import numpy as np

def momentum(start_w, x, y,
                             loss_func,
                             gradient_func,
                             lr=0.01,
                             steps=20,
                             beta=0.9):
    """
    Gradient Descent with Momentum optimizer.

    Parameters:
        start_w : initial weights (numpy array)
        x, y    : input data and targets
        loss_func : function that computes the loss (w, x, y)
        gradient_func : function that computes the gradient (w, x, y)
        lr      : learning rate
        steps   : number of optimization steps
        beta    : momentum coefficient (typically 0.9)

    Returns:
        final_w, loss_history, path_history
    """
    w = start_w.copy()
    v = np.zeros_like(w)  # velocity term

    loss_history = []
    path_history = [w.copy()]

    for _ in range(steps):
        # Compute and record loss
        current_loss = loss_func(w, x, y)
        loss_history.append(current_loss)

        # Compute gradient
        grad = gradient_func(w, x, y)

        # Update velocity and weights
        v = beta * v + grad
        w = w - lr * v

        # Save path for visualization
        path_history.append(w.copy())

    return w, loss_history, np.array(path_history)