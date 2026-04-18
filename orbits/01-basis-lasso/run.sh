#!/usr/bin/env bash
# Reproduce the orbit from scratch:
#   1. run the evaluator on seeds 1/2/3 (test set is deterministic)
#   2. regenerate both figures from the fitted model
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

echo "[1/2] evaluating solution on 3 seeds ..."
for SEED in 1 2 3; do
  python3 research/eval/evaluator.py \
    --solution orbits/01-basis-lasso/solution.py --seed "$SEED"
done

echo "[2/2] regenerating figures ..."
python3 orbits/01-basis-lasso/make_figures.py
