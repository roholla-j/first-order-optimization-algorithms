import numpy as np
from scipy.special import expit

def add_bias(X):
    """
    Appends a column of 1s to the dataset X.
    This corresponds to x_hat = [x, 1]^T, allowing the bias 'b'
    to be absorbed into the weight vector 'w'.
    """
    N = X.shape[0]
    ones = np.ones((N, 1))
    return np.hstack((X, ones))

def logistic_loss(w, X, y):
    """
    Computes the logistic loss for binary classification.
    """
    # Calculate the margin: y_i * (w^T x_i)
    margin = y * (X @ w)

    # np.logaddexp(0, -margin) safely computes log(1 + exp(-margin))
    # without overflowing on large negative margins.
    loss = np.mean(np.logaddexp(0, -margin))
    return loss

def logistic_gradient(w, X, y):
    """
    Computes the full batch gradient of the logistic loss.
    """
    N = X.shape[0]
    margin = y * (X @ w)

    # expit(-margin) = 1/(1+exp(margin)), computed without overflow
    coefficients = -y * expit(-margin)



    # Multiply features by coefficients and average over all N samples
    grad = (X.T @ coefficients) / N
    return grad