# First-Order Optimization Algorithms

A benchmarking framework for comparing first-order gradient-based optimization algorithms on supervised learning tasks and analytical functions. Built as part of a degree project investigating convergence speed, robustness, and hyperparameter sensitivity.

## Algorithms

| Optimizer | Description |
|-----------|-------------|
| **GD** | Full-batch Gradient Descent |
| **SGD** | Stochastic Gradient Descent (mini-batch) |
| **Momentum** | Gradient Descent with heavy-ball momentum |
| **NAG** | Nesterov Accelerated Gradient |
| **Adam** | Adaptive Moment Estimation |

## Benchmark Functions / Datasets

| Problem | Description |
|---------|-------------|
| **Logistic (Australian)** | Binary classification on `australian_scale` (~690 samples) |
| **Logistic (RCV1)** | Binary classification on `rcv1_train.binary` (~23k samples) |
| **Rosenbrock** | Non-convex 2-D function, used for trajectory and convergence analysis |
| **Quadratic Bowl** | Convex 2-D quadratic, used as a sanity-check baseline |

## Metrics

- **Learning rate sweep** — loss curves across a grid of learning rates (all optimizers)
- **Batch size sweep** — loss curves across batch sizes (SGD-style optimizers)
- **Parameter sweep** — single-hyperparameter sweeps (beta for Momentum/NAG, beta1/beta2 for Adam)
- **Sensitivity heatmap** — 2-D grid of final losses over lr × beta, reported with a coefficient-of-variation sensitivity score

## Project Structure

```
├── data/                       # Datasets in LibSVM format
│   ├── australian_scale.txt
│   └── rcv1_train.binary
├── losses/
│   ├── logistic.py             # Logistic loss and gradient
│   ├── quadratic.py            # Quadratic bowl loss and gradient
│   └── rosenbrock.py           # Rosenbrock function and gradient
├── metrics/
│   ├── lr_comparison.py        # Full-batch learning rate sweep
│   ├── sgd_comparison.py       # SGD batch-size × learning-rate sweep
│   ├── param_sweep.py          # Single-parameter sweep (beta, beta1, beta2)
│   ├── hyperparam_comparison.py# 2-D hyperparameter sweep
│   └── sensitivity.py          # Sensitivity heatmap and score
├── optimizers/
│   ├── gd.py
│   ├── sgd.py
│   ├── momentum.py
│   ├── nag.py
│   └── adam.py
├── tests/
│   ├── logistic.py             # Run / plot all logistic regression benchmarks
│   ├── rosenbrock.py           # Run / plot all Rosenbrock benchmarks
│   └── quadratic.py            # Run / plot all quadratic benchmarks
├── utils/
│   ├── config.py               # TOML config loader
│   ├── data.py                 # LibSVM data loader
│   ├── plotting.py             # Visualisation helpers
│   └── runner.py               # Uniform optimizer interface
├── plots/                      # Generated PDF plots (git-tracked)
├── main.py                     # Interactive CLI entry point
└── params.toml                 # All experiment hyperparameters
```

## Running the Benchmarks

The interactive CLI lets you pick a dataset and algorithm:

```bash
python main.py
```

You will be prompted to choose a dataset/function and an algorithm (or "Compare All"). Generated plots are saved to `plots/` as PDF files.

## Configuration

All hyperparameters live in [params.toml](params.toml). The file is structured as:

```toml
[<dataset>]                         # dataset-level settings (path, n_iters)
[<dataset>.<alg>]                   # sweep grids for that algorithm
[<dataset>.compare_all.<alg>]       # best config used in the "Compare All" view
```

Supported dataset keys: `logistic_australian`, `logistic_rcv1`, `quadratic`, `rosenbrock`.

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- tqdm
