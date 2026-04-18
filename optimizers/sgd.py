import numpy as np

def sgd(start_w, x, y,
        loss_func,
        gradient_func,
        lr=0.01,
        steps=100,
        batch_size=1,
        seed=42):
    """
    Stochastic Gradient Descent (SGD) - Modular version
    
    Parameters:
        start_w : initial weights (numpy array)
        x, y    : input data and targets 
                  (for quadratic: x = A, y = b; for datasets: x = features, y = labels)
        loss_func : function that computes the loss (w, x, y)
        gradient_func : function that computes the gradient (w, x, y)
        lr      : learning rate (η)
        steps   : number of parameter updates (not full epochs over data)
        batch_size : number of samples per update (1 = pure SGD)
        seed    : random seed for reproducibility
    
    Returns:
        final_w, loss_history, path_history
    """
    np.random.seed(seed)
    w = start_w.copy()
    loss_history = []
    path_history = [w.copy()]
    
    n = len(x) if hasattr(x, '__len__') and not isinstance(x, (float, int)) else 1
    
    for step in range(steps):
        loss_history.append(loss_func(w, x, y))
        # Sample a mini-batch (true SGD behavior)
        if n > 1:
            indices = np.random.choice(n, batch_size, replace=False)
            x_batch = x[indices]
            y_batch = y[indices]
        else:
            x_batch = x
            y_batch = y

        # Compute gradient on the mini-batch
        grad = gradient_func(w, x_batch, y_batch)
        
        # SGD update: w ← w - η * ∇f_i(w)
        w = w - lr * grad
        
        path_history.append(w.copy())
    
    return w, loss_history, np.array(path_history)