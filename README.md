# First-Order Optimization Algorithms

A benchmarking framework for comparing first-order gradient-based optimization algorithms on supervised learning tasks. Built as part of a degree project investigating convergence speed, robustness, and hyperparameter sensitivity.

## Algorithms

| Optimizer | Description |
|-----------|-------------|
| **GD** | Full-batch Gradient Descent |
| **SGD** | Stochastic Gradient Descent (mini-batch) |
| **Momentum** | Gradient Descent with heavy-ball momentum |
| **NAG** | Nesterov Accelerated Gradient |
| **Adam** | Adaptive Moment Estimation |

## Metrics

- **Convergence speed** — loss curves, area under the loss curve (AUC), and iterations to reach a target loss threshold
- **Robustness** — variance in final loss over 20 random initialisations; degradation under increasing input noise
- **Hyperparameter sensitivity** — final loss across a grid of learning rates (all optimizers) and momentum coefficients (Momentum, NAG)

## Project Structure

```
├── data/                   # Datasets in LibSVM format
├── experiments/
│   ├── benchmark_logistic.py   # Full benchmark on logistic regression
│   └── benchmark_rosenbrock.py # Trajectory benchmark on Rosenbrock function
├── losses/
│   ├── logistic.py         # Logistic loss and gradient
│   └── rosenbrock.py       # Rosenbrock function and gradient
├── metrics/
│   ├── convergence.py      # Convergence speed measurement
│   ├── robustness.py       # Init and noise robustness
│   └── sensitivity.py      # Hyperparameter sensitivity sweep
├── optimizers/
│   ├── gd.py
│   ├── sgd.py
│   ├── momentum.py
│   ├── nag.py
│   └── adam.py
└── utils/
    ├── data.py             # LibSVM data loader
    ├── plotting.py         # Visualisation functions
    └── runner.py           # Uniform optimizer interface
```

## Running the Benchmarks

```bash
# Logistic regression benchmark (convergence, robustness, sensitivity)
python experiments/benchmark_logistic.py

# Rosenbrock benchmark (loss curves, trajectories, distance to optimum)
python experiments/benchmark_rosenbrock.py
```

To switch datasets, change the path in `benchmark_logistic.py`:

```python
X, y = load_data("data/australian_scale.txt")   # small dataset (~690 samples)
X, y = load_data("data/rcv1_train.binary")      # large dataset (~23k samples)
```

Both files must be in LibSVM format.

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib