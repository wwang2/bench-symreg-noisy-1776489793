#!/usr/bin/env python3
"""
Generate noisy data for symbolic regression benchmark.

Target function (HIDDEN from agents):
  f(x) = 0.5 * sin(2x) * exp(-0.1 * x^2) + 0.3 * x * cos(x) - 0.2

This is a composed function with:
- Damped oscillation: sin(2x) * exp(-0.1x^2)
- Phase-shifted product: x * cos(x)
- Constant offset

Agents receive ONLY the data points, not the formula.
"""

import numpy as np
import os

def target_function(x):
    """The true function (not revealed to agents)."""
    return 0.5 * np.sin(2 * x) * np.exp(-0.1 * x**2) + 0.3 * x * np.cos(x) - 0.2

def generate_train_data(n_points=50, noise_sigma=0.05, seed=42):
    """Generate noisy training data."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-5, 5, n_points)
    y_true = target_function(x)
    y_noisy = y_true + rng.normal(0, noise_sigma, n_points)
    return x, y_noisy

def generate_test_data(n_points=500, seed=99):
    """Generate clean test data for evaluation (no noise)."""
    x = np.linspace(-5, 5, n_points)
    y = target_function(x)
    return x, y

if __name__ == "__main__":
    # Generate and save training data
    x_train, y_train = generate_train_data()
    np.savetxt("train_data.csv", np.column_stack([x_train, y_train]),
               delimiter=",", header="x,y", comments="")
    print(f"Training data: {len(x_train)} points, noise=0.05")
    print(f"  x range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    # Generate test data (for evaluator)
    x_test, y_test = generate_test_data()
    np.savetxt("test_data.csv", np.column_stack([x_test, y_test]),
               delimiter=",", header="x,y", comments="")
    print(f"Test data: {len(x_test)} points (clean, for evaluation)")
