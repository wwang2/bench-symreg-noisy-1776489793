"""Symbolic regression via rich basis library + LASSO cross-validation.

Strategy:
  1. Build a rich library of candidate basis functions:
     - Polynomials x^k, k=0..10
     - Trigonometric sin(k*x), cos(k*x) for k=0.25..3 (several frequencies)
     - Damped Gaussians exp(-(x-mu)^2 / sigma^2) centered on grid
     - Products x*sin(k*x), x*cos(k*x)
     - Simple nonlinear features: tanh(a*x), 1/(1+x^2)
  2. Standardize features, center y.
  3. Fit LassoCV with 5-fold CV to select regularization strength.
  4. Return the learned linear combination as f(x).

We fit once at import time on the training data — the evaluator calls `f(x)` as a
pure function thereafter.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
# Evaluator loads this module from the repo root; prefer the absolute path
# via the worktree structure.
_TRAIN_CSV_CANDIDATES = [
    _HERE.parent.parent / "research" / "eval" / "train_data.csv",
    Path.cwd() / "research" / "eval" / "train_data.csv",
]


def _load_training() -> tuple[np.ndarray, np.ndarray]:
    for p in _TRAIN_CSV_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            return df["x"].to_numpy(), df["y"].to_numpy()
    raise FileNotFoundError(
        f"train_data.csv not found in any of: {_TRAIN_CSV_CANDIDATES}"
    )


# ---- Basis library ----------------------------------------------------------


def _make_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build the rich candidate-function library at points x.

    Returns the design matrix Phi of shape (N, D) and a list of feature names
    aligned with the columns (useful for reporting / figures).
    """
    x = np.asarray(x, dtype=float).ravel()
    feats: list[np.ndarray] = []
    names: list[str] = []

    # Polynomials x^0 .. x^10  (x^0 is a bias column; StandardScaler will drop
    # its variance contribution — we add an explicit intercept in LassoCV).
    for k in range(0, 11):
        feats.append(x ** k)
        names.append(f"x^{k}")

    # Trig basis at a range of frequencies
    freqs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    for w in freqs:
        feats.append(np.sin(w * x))
        names.append(f"sin({w}x)")
        feats.append(np.cos(w * x))
        names.append(f"cos({w}x)")

    # x * sin / cos (modulated oscillations)
    for w in [0.5, 1.0, 1.5]:
        feats.append(x * np.sin(w * x))
        names.append(f"x*sin({w}x)")
        feats.append(x * np.cos(w * x))
        names.append(f"x*cos({w}x)")

    # Damped Gaussians e^{-(x-mu)^2 / (2 sigma^2)}
    for mu in np.linspace(-5.0, 5.0, 11):
        for sigma in [1.0, 2.0]:
            feats.append(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
            names.append(f"gauss({mu:.1f},{sigma})")

    # Logistic / tanh
    for a in [0.5, 1.0, 2.0]:
        feats.append(np.tanh(a * x))
        names.append(f"tanh({a}x)")

    # Rational bumps
    feats.append(1.0 / (1.0 + x ** 2))
    names.append("1/(1+x^2)")
    feats.append(x / (1.0 + x ** 2))
    names.append("x/(1+x^2)")

    Phi = np.column_stack(feats)
    return Phi, names


# ---- Fit (once, at import) --------------------------------------------------


def _fit() -> dict:
    x_tr, y_tr = _load_training()
    Phi_tr, names = _make_features(x_tr)

    # Drop the constant-column (x^0) from the scaler — intercept handled by
    # LassoCV directly.
    keep = np.arange(Phi_tr.shape[1]) != 0  # drop x^0
    Phi_tr_k = Phi_tr[:, keep]
    names_k = [n for n, k in zip(names, keep) if k]

    scaler = StandardScaler()
    Phi_scaled = scaler.fit_transform(Phi_tr_k)

    # LassoCV with 5-fold CV picks alpha. A wide alpha grid on log-scale.
    model = LassoCV(
        cv=5,
        n_alphas=200,
        eps=1e-4,
        max_iter=200_000,
        tol=1e-7,
        fit_intercept=True,
        random_state=0,
        selection="cyclic",
    )
    model.fit(Phi_scaled, y_tr)

    return {
        "scaler": scaler,
        "model": model,
        "keep_mask": keep,
        "names": names,
        "names_kept": names_k,
        "alpha": float(model.alpha_),
        "coef": model.coef_,
        "intercept": float(model.intercept_),
        "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-10)),
        "x_train": x_tr,
        "y_train": y_tr,
    }


_FIT = _fit()


# ---- Public API -------------------------------------------------------------


def f(x: np.ndarray) -> np.ndarray:
    """Predict y for new x using the fitted LASSO model."""
    x = np.asarray(x, dtype=float).ravel()
    Phi, _ = _make_features(x)
    Phi_k = Phi[:, _FIT["keep_mask"]]
    Phi_scaled = _FIT["scaler"].transform(Phi_k)
    return _FIT["model"].predict(Phi_scaled)


# Diagnostic: how sparse did LASSO go?
if __name__ == "__main__":
    print(f"alpha* = {_FIT['alpha']:.6g}")
    print(f"non-zero coeffs: {_FIT['n_nonzero']} / {len(_FIT['names_kept'])}")
    nz = np.argsort(-np.abs(_FIT["coef"]))[: _FIT["n_nonzero"]]
    for i in nz:
        print(f"  {_FIT['names_kept'][i]:20s}  w={_FIT['coef'][i]:+.4f}")
