#!/usr/bin/env python3
"""
Evaluator for symbolic regression benchmark.

Scores a solution's f(x) against held-out test data.
The test data is generated from the true function (hidden from agents).

Usage:
    python evaluator.py --solution <path> --seed <int>
    Output: METRIC=<float>  (MSE on test set)
"""

import argparse
import importlib.util
import numpy as np
import sys
import os

# Import the data generator (co-located)
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_DIR)
from generate_data import generate_test_data


def load_solution(path):
    """Dynamically load a solution module."""
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(solution_path, seed=42):
    """Evaluate solution against test data."""
    try:
        module = load_solution(solution_path)
    except Exception as e:
        print(f"ERROR: Could not load solution: {e}", file=sys.stderr)
        sys.exit(1)

    # Get the solution's prediction function
    if hasattr(module, 'f'):
        predict = module.f
    elif hasattr(module, 'solve'):
        predict = module.solve(seed=seed)
    else:
        print("ERROR: Solution must define f(x) or solve(seed)", file=sys.stderr)
        sys.exit(1)

    # Generate test data (deterministic, same for all seeds)
    x_test, y_test = generate_test_data(n_points=500, seed=99)

    try:
        y_pred = predict(x_test)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
    except Exception as e:
        print(f"ERROR: Solution prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # MSE
    mse = np.mean((y_test - y_pred) ** 2)
    return mse


def main():
    parser = argparse.ArgumentParser(description="Symbolic Regression Evaluator")
    parser.add_argument("--solution", required=True, help="Path to solution.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    metric = evaluate(args.solution, args.seed)
    print(f"METRIC={metric:.10f}")


if __name__ == "__main__":
    main()
