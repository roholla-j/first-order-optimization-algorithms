import numpy as np

def quadratic_loss(w, A, b):
    return (w.T @ A @ w) - 2 * (b.T @ w)

def quadratic_gradient(w, A, b):
    return 2 * (A @ w) - 2 * b
