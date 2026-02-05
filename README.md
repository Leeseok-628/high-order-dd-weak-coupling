# Least-squares DD optimization (project skeleton)

This folder contains:
- `dd/`: helper modules (moments optimization, Hamiltonian/evolver, pulse compilation, metrics/plotting)
- `least_square_dd_optimization.ipynb`: runnable notebook

## Quick start
1) Create an environment and install deps:
   - `pip install -r requirements.txt`

2) Open the notebook and run all cells:
   - `jupyter lab` (or `jupyter notebook`)

The notebook reproduces:
- Moment-cancellation timing optimization for K=1..8
- Weak-coupling scaling check via operator-norm error vs T on a random bath


## QDD comparison
- `qdd_comparison_trace_distance.ipynb`: Monte Carlo trace-distance comparison (Our DD vs QDD)
