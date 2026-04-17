
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
        # Compute and record loss
        current_loss = loss_func(w, x, y)
        loss_history.append(current_loss)
        
        # Update weights using gradient
        grad = gradient_func(w, x, y)
        w = w - lr * grad
        
        # Save path for visualization
        path_history.append(w.copy())

    return w, loss_history, np.array(path_history)