import numpy as np
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sp

def load_data(path: str):
    X, y = load_svmlight_file(path)
    # y = np.where(y > 0, 1, -1)

    # Add bias term as a sparse column
    ones = sp.csr_matrix(np.ones((X.shape[0], 1)))
    X = sp.hstack([X, ones]).tocsr()

    return X, y
