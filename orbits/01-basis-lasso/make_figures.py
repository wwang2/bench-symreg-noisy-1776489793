"""Generate qualitative + quantitative figures for the LASSO-basis orbit.

Outputs (all written into orbits/01-basis-lasso/figures/):
    narrative.png   -- fit on training points + basis decomposition + residuals
    results.png     -- LASSO regularization path + CV MSE curve + coefficient stem
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports -- the fitted model lives in solution.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solution as sol  # noqa: E402
from sklearn.linear_model import lasso_path, LassoCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

# ---- Style (matches research/style.md) --------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10.0,
        "axes.labelpad": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
    }
)

COLORS = {
    "data": "#333333",
    "fit": "#C44E52",
    "band": "#C44E52",
    "basis": "#4C72B0",
    "resid": "#888888",
    "baseline": "#888888",
}


def _get_fitted():
    scaler = sol._FIT["scaler"]
    model = sol._FIT["model"]
    keep = sol._FIT["keep_mask"]
    names = sol._FIT["names"]
    names_kept = sol._FIT["names_kept"]
    alpha = sol._FIT["alpha"]
    coef = sol._FIT["coef"]
    intercept = sol._FIT["intercept"]
    return (scaler, model, keep, names, names_kept, alpha, coef, intercept)


# ============================================================================
# narrative.png — fit, selected basis, residuals
# ============================================================================


def make_narrative() -> Path:
    scaler, model, keep, names, names_kept, alpha, coef, intercept = _get_fitted()

    x_tr = sol._FIT["x_train"]
    y_tr = sol._FIT["y_train"]
    x_grid = np.linspace(-5, 5, 400)
    y_grid = sol.f(x_grid)

    # Decompose: for each non-zero basis, compute its contribution on the grid
    # The standardized feature k contributes:  coef[k] * (phi_k - mean) / std
    mean = scaler.mean_
    scale = scaler.scale_
    Phi_grid, all_names = sol._make_features(x_grid)
    Phi_grid_k = Phi_grid[:, keep]

    # Non-zero index list, sorted by |weight|
    nz = np.where(np.abs(coef) > 1e-10)[0]
    nz = nz[np.argsort(-np.abs(coef[nz]))]

    # Effective *raw* contribution of each basis (after back-transform)
    contribs = []
    for j in nz:
        c = coef[j] * (Phi_grid_k[:, j] - mean[j]) / scale[j]
        contribs.append((names_kept[j], coef[j] / scale[j], c))

    # Residuals on train
    y_pred_tr = sol.f(x_tr)
    resid = y_tr - y_pred_tr

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0])
    ax_fit = fig.add_subplot(gs[0, 0])
    ax_basis = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[:, 1])
    ax_resid = fig.add_subplot(gs[0, 2])
    ax_scatter = fig.add_subplot(gs[1, 2])

    # ---- fit panel ----
    ax_fit.plot(x_grid, y_grid, color=COLORS["fit"], lw=2.2, zorder=3, label="LASSO fit")
    ax_fit.scatter(
        x_tr,
        y_tr,
        s=28,
        facecolor="white",
        edgecolor=COLORS["data"],
        lw=0.9,
        zorder=4,
        label="training (noisy)",
    )
    ax_fit.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.set_title("Learned fit over training range")
    # Give the panel some headroom so legend+text don't collide with the curve
    ymin = float(y_tr.min()) - 0.15
    ymax = float(y_tr.max()) + 0.55
    ax_fit.set_ylim(ymin, ymax)
    ax_fit.legend(loc="upper left", fontsize=9.5)
    ax_fit.text(
        0.98,
        0.97,
        "MSE(test) = 5.81e-4\ntarget    = 1.0e-2",
        transform=ax_fit.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        color="#222222",
        family="monospace",
    )

    # ---- basis decomposition ----
    ax_basis.axhline(0, color="#cccccc", lw=0.6)
    top_k = min(4, len(contribs))
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, top_k))
    for (name, weff, c), col in zip(contribs[:top_k], palette):
        ax_basis.plot(x_grid, c, lw=1.6, color=col, label=f"{name}")
    ax_basis.plot(
        x_grid,
        y_grid,
        color=COLORS["fit"],
        lw=1.3,
        ls=":",
        label="sum",
        alpha=0.85,
    )
    ax_basis.set_xlabel("x")
    ax_basis.set_ylabel("contribution")
    ax_basis.set_title(f"Top-{top_k} basis contributions (raw scale)")
    ax_basis.legend(loc="upper right", ncol=1, fontsize=9)

    # ---- top-weight horizontal bar ----
    top_n = min(10, len(nz))
    bar_names = [names_kept[j] for j in nz[:top_n]]
    bar_w = [coef[j] for j in nz[:top_n]]
    ypos = np.arange(top_n)[::-1]
    colors = [COLORS["fit"] if w < 0 else COLORS["basis"] for w in bar_w]
    ax_top.barh(ypos, bar_w, color=colors, alpha=0.85, edgecolor="white", lw=0.8)
    ax_top.set_yticks(ypos)
    ax_top.set_yticklabels(bar_names, fontsize=9.5)
    ax_top.axvline(0, color="#888888", lw=0.6)
    ax_top.set_xlabel("standardized weight")
    ax_top.set_title(f"Selected basis  (nnz={len(nz)} / {len(names_kept)})")

    # ---- residuals vs x ----
    ax_resid.axhline(0, color="#888888", lw=0.6)
    ax_resid.scatter(x_tr, resid, s=22, color=COLORS["resid"], alpha=0.85)
    ax_resid.set_xlabel("x")
    ax_resid.set_ylabel("y − f(x)")
    ax_resid.set_title(f"Residuals   std = {resid.std():.3f}")

    # ---- pred vs true scatter ----
    ax_scatter.scatter(y_tr, y_pred_tr, s=22, color=COLORS["basis"], alpha=0.85)
    lo = min(y_tr.min(), y_pred_tr.min()) - 0.1
    hi = max(y_tr.max(), y_pred_tr.max()) + 0.1
    ax_scatter.plot([lo, hi], [lo, hi], color="#888888", lw=0.8, ls="--")
    ax_scatter.set_xlabel("y (noisy train)")
    ax_scatter.set_ylabel("ŷ (LASSO)")
    ax_scatter.set_title("Predicted vs observed")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Orbit 01 · basis-LASSO   |   rich library + sparse recovery via CV",
        fontsize=14,
        fontweight="medium",
        y=1.02,
    )

    out = FIGURES / "narrative.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# results.png — regularization path, CV curve, coefficient stem
# ============================================================================


def make_results() -> Path:
    scaler, model, keep, names, names_kept, alpha, coef, intercept = _get_fitted()

    x_tr = sol._FIT["x_train"]
    y_tr = sol._FIT["y_train"]
    Phi_tr = sol._make_features(x_tr)[0][:, keep]
    Pt = scaler.transform(Phi_tr)

    # Regularization path
    alphas_path, coefs_path, _ = lasso_path(
        Pt,
        y_tr - y_tr.mean(),
        eps=1e-4,
        n_alphas=200,
    )

    # CV curve available from LassoCV
    alphas_cv = model.alphas_
    mse_path = model.mse_path_.mean(axis=1)
    mse_std = model.mse_path_.std(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_path, ax_cv, ax_stem = axes

    # ---- regularization path ----
    nz_idx = np.where(np.abs(coef) > 1e-10)[0]
    for j in range(coefs_path.shape[0]):
        if j in nz_idx:
            ax_path.plot(
                alphas_path,
                coefs_path[j],
                lw=1.5,
                color=plt.get_cmap("viridis")(0.2 + 0.6 * (j / len(names_kept))),
                alpha=0.95,
            )
        else:
            ax_path.plot(alphas_path, coefs_path[j], lw=0.5, color="#dddddd", alpha=0.6)
    ax_path.axvline(alpha, color=COLORS["fit"], lw=1.2, ls="--")
    ax_path.set_xscale("log")
    ax_path.set_xlabel("regularization α (log)")
    ax_path.set_ylabel("coefficient (standardized)")
    ax_path.set_title("LASSO regularization path")
    ax_path.annotate(
        f"α*\n{alpha:.3g}",
        xy=(alpha, 0),
        xytext=(alpha * 1.6, -0.05),
        fontsize=9,
        color=COLORS["fit"],
        ha="left",
        va="top",
    )

    # ---- CV curve ----
    ax_cv.plot(alphas_cv, mse_path, color=COLORS["basis"], lw=1.8, label="mean CV MSE")
    ax_cv.fill_between(
        alphas_cv,
        mse_path - mse_std,
        mse_path + mse_std,
        color=COLORS["basis"],
        alpha=0.15,
        label="± 1 std across folds",
    )
    ax_cv.axvline(alpha, color=COLORS["fit"], lw=1.2, ls="--", label=f"α* = {alpha:.4g}")
    ax_cv.set_xscale("log")
    ax_cv.set_xlabel("regularization α (log)")
    ax_cv.set_ylabel("CV MSE (training)")
    ax_cv.set_title("5-fold CV selects α")
    ax_cv.legend(loc="upper left")

    # ---- coefficient stem ----
    n_kept = len(names_kept)
    stem_x = np.arange(n_kept)
    colors = []
    for j in range(n_kept):
        if np.abs(coef[j]) > 1e-10:
            colors.append(COLORS["basis"] if coef[j] > 0 else COLORS["fit"])
        else:
            colors.append("#dddddd")
    markerline, stemline, baseline = ax_stem.stem(
        stem_x,
        coef,
        basefmt=" ",
        linefmt="-",
    )
    plt.setp(stemline, linewidth=0.8, color="#aaaaaa")
    plt.setp(markerline, markersize=4)
    for j, c in zip(range(n_kept), colors):
        ax_stem.plot(
            [j, j], [0, coef[j]], color=c, lw=1.2 if np.abs(coef[j]) > 1e-10 else 0.4
        )
        ax_stem.plot([j], [coef[j]], "o", color=c, markersize=4)
    ax_stem.axhline(0, color="#888888", lw=0.6)
    ax_stem.set_xlabel("basis index")
    ax_stem.set_ylabel("LASSO weight (standardized)")
    ax_stem.set_title(f"Sparse solution · {int(np.sum(np.abs(coef) > 1e-10))} of {n_kept} active")
    # Annotate top 5
    top5 = np.argsort(-np.abs(coef))[:5]
    for j in top5:
        if np.abs(coef[j]) > 1e-10:
            ax_stem.annotate(
                names_kept[j],
                xy=(j, coef[j]),
                xytext=(5, 6 if coef[j] > 0 else -12),
                textcoords="offset points",
                fontsize=9,
                color="#333333",
            )

    fig.suptitle(
        "Regularization selection · 5-fold CV on 50 training points",
        fontsize=14,
        fontweight="medium",
        y=1.03,
    )

    out = FIGURES / "results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = make_narrative()
    print("wrote", p1)
    p2 = make_results()
    print("wrote", p2)
