
import numpy as np
from losses.logistic import add_bias

def load_libsvm(path: str):
    X_rows, y_list = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            y_list.append(float(parts[0]))
            pairs = {int(k): float(v) for k, v in (p.split(':') for p in parts[1:])}
            X_rows.append(pairs)

    n_features = max(k for row in X_rows for k in row)
    X = np.zeros((len(X_rows), n_features))
    for i, row in enumerate(X_rows):
        for j, v in row.items():
            X[i, j - 1] = v

    y = np.array(y_list)
    return X, y

def load_data(path: str):
    X, y = load_libsvm(path)
    y = np.where(y == 0, -1, 1)
    X = add_bias(X)
    return X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]