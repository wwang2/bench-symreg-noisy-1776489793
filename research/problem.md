# Symbolic Regression (Noisy)

## Problem Statement
Find the function f(x) that produced the data points in `research/eval/train_data.csv`. The training data is noisy. A hidden test set is used to score predictions.

## Solution Interface
Solution must be a Python file at `orbits/<name>/solution.py` exposing either:
- `f(x: np.ndarray) -> np.ndarray`, or
- `solve(seed: int) -> Callable[[np.ndarray], np.ndarray]`

The evaluator (`research/eval/evaluator.py`) prefers `f(x)` if present.

## Success Metric
MSE on held-out test set (minimize). Target: 0.01.

## Budget
Max 3 orbits.
