---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.000581
---

# basis-LASSO — rich library + sparse recovery

**Result:** MSE = 5.81e-4 on held-out test (target = 1.0e-2). Seventeen times
below target, deterministic across seeds.

| Seed | Metric (MSE) | Time |
|------|--------------|------|
| 1    | 0.000581     | 13 s |
| 2    | 0.000581     | 13 s |
| 3    | 0.000581     | 13 s |
| **Mean** | **0.000581 ± 0.000000** | |

(Seeds are identical because the evaluator always uses the *same* test set —
`generate_test_data(seed=99)` — and the model is fully determined by the 50
training points; the `--seed` argument does not perturb anything downstream of
our `f`.)

## The idea in one sentence

Pick a dense dictionary of candidate basis functions, standardize them, and let
5-fold-CV LASSO decide which ones survive. The regularizer does the symbolic
selection automatically.

## Why this should work before we run it

The training data is 50 noisy samples on a smooth function over `x ∈ [-5, 5]`.
A smooth scalar function on a bounded interval can be written as a linear
combination of a handful of well-chosen basis functions — polynomials capture
the low-frequency envelope, sines and cosines capture the oscillations,
Gaussians handle localized features. The only question is *which* combination,
and with only 50 data points we have a high risk of overfitting the noise if we
use unregularized least squares. LASSO's `ℓ₁` penalty forces most coefficients
to zero, so we can afford to hand it a rich (over-complete) dictionary and let
cross-validation pick the regularization strength. If the true function is
sparse in our dictionary, LASSO recovers it up to noise; if it is not, we
still get the best `ℓ₁`-regularized linear projection.

## The dictionary

61 non-constant basis functions (plus intercept handled by LASSO):

| Family                           | Functions | Count |
|----------------------------------|-----------|------:|
| Polynomials                      | x, x², …, x¹⁰ | 10 |
| Trigonometric                    | sin(ωx), cos(ωx), ω ∈ {0.25, 0.5, …, 3.0} | 18 |
| Modulated trigonometric          | x·sin(ωx), x·cos(ωx), ω ∈ {0.5, 1.0, 1.5} | 6 |
| Gaussian bumps                   | exp(−½(x−μ)²/σ²), μ ∈ {−5, −4, …, 5}, σ ∈ {1, 2} | 22 |
| Sigmoidal / rational             | tanh(½x), tanh(x), tanh(2x), 1/(1+x²), x/(1+x²) | 5 |

Features are standardized (zero mean, unit variance) *before* fitting so the
`ℓ₁` penalty treats them comparably. `LassoCV(cv=5, n_alphas=200, eps=1e-4)`
selects `α*` on a 200-point log grid.

## What LASSO found

The CV picks `α* = 5.99 × 10⁻³`, and keeps 9 of 61 features:

| basis          | standardized weight |
|----------------|--------------------:|
| x · cos(1.0 x) |             +0.5003 |
| sin(2.0 x)     |             +0.1315 |
| sin(1.5 x)     |             +0.1028 |
| sin(2.5 x)     |             +0.0867 |
| gauss(−4, 1)   |             +0.0187 |
| sin(1.25 x)    |             +0.0143 |
| gauss( 3, 1)   |             −0.0110 |
| cos(3.0 x)     |             −0.0031 |
| cos(2.5 x)     |             −0.0028 |

The dominant term is `x·cos(x)`: a slow-envelope oscillation that explains the
bulk of the structure. The additional mid-to-high-frequency sines are the
residual detail that distinguishes the true function from a pure `x·cos(x)`
(baseline: fitting only `a·x·cos(x) + b` on this data gives MSE 0.048 on the
held-out set — two orders of magnitude worse than the sparse LASSO).

## Sanity checks I ran

- **Does the evaluator care about `--seed`?** No. The test set is deterministic
  (`seed=99` hard-coded inside `evaluator.py`) and our `f` is deterministic,
  so all three seeds give identical MSE. Reported the mean anyway because
  that is the campaign convention.
- **Could a simpler model suffice?** `y = a·x·cos(x) + b` alone gets MSE
  0.048 — fails the 0.01 target by 5×. Adding `sin(2x)` brings it to 0.008.
  LASSO's nine-term sparse combination reaches 0.00058 without us having to
  guess the right three terms up front.
- **Is the noise floor binding?** The residual std on the 50-point training
  data is ≈ 0.043 (`y − f(x)`). Since the test set is *clean*, the true
  function sits inside our 9-term sparse combination up to an MSE of ~6e-4,
  meaning our dictionary is expressive enough that the limiting factor was
  which features to pick, not the dictionary itself.

## Prior Art & Novelty

### What is already known

- **LASSO** ([Tibshirani, 1996](https://www.jstor.org/stable/2346178)) — `ℓ₁`
  regularized least squares; the canonical tool for sparse linear model
  selection.
- **Basis-pursuit / dictionary learning for function approximation** — the
  idea of over-complete libraries + sparse recovery is standard compressed
  sensing (Candès, Donoho, Tao, 2006).
- **SINDy** ([Brunton, Proctor, Kutz, 2016](https://arxiv.org/abs/1509.03580))
  — applies exactly this recipe (library of candidate terms + sparse
  regression) to identify governing equations of dynamical systems. The
  numerical approach is identical; we apply it to a static `y = f(x)` fit
  rather than a derivative-matching problem.
- **Symbolic regression** (Koza, 1992; Schmidt & Lipson, 2009) — the genetic-
  programming alternative. Slower and more exploratory, but discovers
  nonlinear compositions outside a fixed basis.

### What this orbit adds

Nothing novel; this is a textbook application of LASSO for basis-selection.
Its value in this campaign is as a **strong baseline**: any future orbit
(genetic symbolic regression, transformer pre-trained on numerical data,
Bayesian model selection over a basis) must beat MSE = 5.81e-4, which is
already 17× below the campaign target.

### Honest positioning

LASSO returns a *functional form* (a linear combination of dictionary
elements) rather than a compact symbolic expression. If the ground-truth
formula is `y = x · cos(x)·<something>` it is hiding inside our 9 selected
terms — but reading the formula off the bar chart is a human-in-the-loop
step, not something the pipeline does. A SINDy-style thresholded least squares
pass (drop small weights, refit unregularized) would recover a cleaner
expression at the cost of a slightly larger MSE.

## Failure modes to watch

- If the true function contains features *outside* the dictionary (e.g.
  `log(x+k)`, `|x|`, a discontinuity), LASSO silently produces the best
  projection and the residual variance will not drop. The Gaussian-bump rows
  in our library are a partial hedge.
- CV with only 50 points has non-trivial variance. Running `LassoCV` with a
  different `random_state` picks slightly different `α*` values but the
  selected features and MSE move by <5 %.

## Glossary

- **LASSO** — Least Absolute Shrinkage and Selection Operator; linear
  regression with an `ℓ₁` penalty on coefficients.
- **CV** — cross-validation; here, 5-fold.
- **MSE** — mean squared error.
- **nnz** — number of non-zero coefficients.
- **SINDy** — Sparse Identification of Nonlinear Dynamics.

## Files

- `solution.py` — fits at import, exposes `f(x)` (pure function after fit).
- `make_figures.py` — regenerates the figures from the fitted model.
- `figures/narrative.png` — fit on training points, top basis functions, residuals, predicted-vs-observed.
- `figures/results.png` — regularization path, 5-fold CV curve, sparse coefficient stem plot.

## References

- Tibshirani (1996), *Regression Shrinkage and Selection via the Lasso*, JRSS-B.
- Brunton, Proctor & Kutz (2016), [*Discovering governing equations from data by sparse identification of nonlinear dynamical systems*](https://arxiv.org/abs/1509.03580), PNAS.
- Candès & Tao (2006), *Near-optimal signal recovery from random projections: Universal encoding strategies?*, IEEE-IT.
