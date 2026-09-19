"""
figures/smoothing.py - Smooth error rate against signed illusion strength.

Follows the modelling approach of Makowski et al. (2023), who fitted additive
models per illusion to error rate and reaction time directly rather than to any
derived threshold. Here: a natural cubic regression spline in signed illusion
strength, one curve per (species, illusion), fitted by binomial GLM on the
aggregated response counts.

TWO CHOICES WORTH KNOWING ABOUT
-------------------------------
Degrees of freedom are FIXED at SPLINE_DF rather than selected per curve.
Selection by BIC was tried and rejected: with up to 2048 trials per cell the
binomial likelihood is so confident that BIC keeps buying flexibility until the
spline nearly interpolates the 15 tested levels, which is the jaggedness the
smooth exists to remove. A fixed modest basis is also comparable across panels.
Among 5, 6 and 7, only 6 fits every curve without degenerating (a df of 5
cannot follow the model's step at zero strength on Delboeuf, and its residual
dispersion explodes to ~6300).

The ribbon is QUASI-BINOMIAL. Residual dispersion on these cells runs 23-40,
because each cell pools participants who differ in susceptibility, so a plain
binomial interval is roughly five times too narrow and would imply precision
the design cannot support. The variance is therefore scaled by the Pearson
dispersion estimate. This is a correction for the aggregation, not a substitute
for the participant random effects that Makowski et al. could fit on
trial-level data; with per-participant counts the right model would carry a
random slope per participant instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.special import expit

# Spline bases per curve. See the module docstring for why this is fixed.
SPLINE_DF = 6

# Points across the strength axis at which the fitted curve is evaluated.
N_GRID = 240

_Z95 = 1.959964


def _basis(design_info, grid: np.ndarray) -> np.ndarray:
    """The fitted spline basis evaluated at `grid`."""
    return build_design_matrices(
        [design_info], {"x": grid}, return_type="dataframe"
    )[0].to_numpy()


def _fit_error_rate(cells: pd.DataFrame, df: int):
    """
    The binomial spline fit behind smooth_error_rate.

    Returns (design_info, beta, cov), with the dispersion correction already
    in `cov`, or None when there are too few levels to support the basis.
    """
    sub = cells.sort_values("illusion_strength")
    if len(sub) <= df:
        return None

    smax = float(sub["illusion_strength"].abs().max())
    if not np.isfinite(smax) or smax <= 0:
        return None

    x = (sub["illusion_strength"] / smax).to_numpy(float)
    k = sub["n_errors"].to_numpy(float)
    n = sub["n_trials"].to_numpy(float)

    design = dmatrix(f"cr(x, df={df}) - 1", {"x": x}, return_type="dataframe")
    model = sm.GLM(
        np.column_stack([k, n - k]), design, family=sm.families.Binomial()
    )
    res = model.fit()

    # Pearson dispersion, floored at 1 so the correction only ever widens.
    dispersion = max(float(res.pearson_chi2 / res.df_resid), 1.0)
    return (
        design.design_info,
        res.params.to_numpy(),
        res.cov_params().to_numpy() * dispersion,
    )


def smooth_error_rate(
    cells: pd.DataFrame,
    df: int = SPLINE_DF,
    n_grid: int = N_GRID,
) -> pd.DataFrame | None:
    """
    Fit one smooth of error rate against normalised signed illusion strength.

    Args:
        cells:   rows for a single (species, illusion), carrying
                 `illusion_strength`, `n_errors` and `n_trials`.
        df:      spline bases.
        n_grid:  points at which to evaluate the fitted curve.

    Returns:
        A frame with `x` in [-1, 1] and `fit`, `lo`, `hi` as percentages, or
        None when there are too few levels to support the basis.
    """
    fit = _fit_error_rate(cells, df)
    if fit is None:
        return None
    design_info, beta, cov = fit

    grid = np.linspace(-1.0, 1.0, n_grid)
    basis = _basis(design_info, grid)
    eta = basis @ beta
    # Diagonal of basis @ cov @ basis.T without forming the full matrix.
    var_eta = np.einsum("ij,jk,ik->i", basis, cov, basis)

    # The band is carried to the percentage scale by the delta method rather
    # than by transforming the log-odds interval. Near 0% errors - where the
    # model sits on the congruent side of most illusions - a log-odds interval
    # is dominated by how poorly the spline's end is pinned in log-odds units,
    # and back-transformed it flared to 70% on Muller-Lyer while the curve sat
    # at 0.1% and the data showed 1 error in 880 trials. On the percentage scale
    # the width follows p(1 - p), so it narrows where errors are rare, as the
    # counts themselves do.
    p = expit(eta)
    half = _Z95 * p * (1.0 - p) * np.sqrt(np.clip(var_eta, 0.0, None))

    return pd.DataFrame(
        {
            "x": grid,
            "fit": p * 100.0,
            "lo": np.clip(p - half, 0.0, 1.0) * 100.0,
            "hi": np.clip(p + half, 0.0, 1.0) * 100.0,
        }
    )


def error_effect(
    cells: pd.DataFrame,
    df: int = SPLINE_DF,
    n_grid: int = N_GRID,
) -> tuple[float, float] | None:
    """
    The illusion's effect on error rate, from the same fit as the drawn curve.

    The smoothed error rate averaged over incongruent strengths (0, 1] minus
    its average over congruent strengths [-1, 0), in percentage points. The
    difference removes the baseline error level, which depends on task
    conditions (speeded humans, untimed model) rather than on the illusion.

    The variance is the delta method on the spline coefficients, with the
    same dispersion correction as the drawn band. The gradient of a mean of
    expit(basis @ beta) is the mean of p(1 - p) * basis, so no resampling is
    needed.

    Returns:
        (effect, variance), or None when the curve cannot be fitted.
    """
    fit = _fit_error_rate(cells, df)
    if fit is None:
        return None
    design_info, beta, cov = fit

    grid = np.linspace(-1.0, 1.0, n_grid)
    basis = _basis(design_info, grid)
    p = expit(basis @ beta)
    slope = (p * (1.0 - p))[:, None] * basis
    incongruent, congruent = grid > 0, grid < 0

    effect = 100.0 * (p[incongruent].mean() - p[congruent].mean())
    grad = 100.0 * (slope[incongruent].mean(0) - slope[congruent].mean(0))
    return float(effect), float(grad @ cov @ grad)


# Spline bases for the magnitude smooth. Stiffer than the error-rate smooth:
# each magnitude curve has only seven independent levels (the other half of the
# signed axis is their mirror), and at six bases the curves followed level-to-
# level variation that reads as wiggle rather than trend.
MAGNITUDE_DF = 4


def smooth_magnitude(
    levels: pd.DataFrame,
    df: int = MAGNITUDE_DF,
    n_grid: int = N_GRID,
) -> pd.DataFrame | None:
    """
    Smooth illusion magnitude against signed strength, the same way Figure 3 is.

    The same natural cubic regression spline as the error-rate smooth, with a
    smaller basis (MAGNITUDE_DF), fitted across the full signed axis. A natural spline is
    local and turns linear at its boundaries, which is what keeps the curve
    from overshooting at the ends of the tested range. A single global cubic
    was tried first and rejected for exactly that: on Delboeuf it dipped and
    rebounded past the last two levels, which were in fact still falling, and
    on Ponzo one steep endpoint bent the whole human curve.

    Each magnitude level is mirrored to its negative strength, so the curve is
    odd in the illusion's spatial direction, and the origin is included as a
    level of its own - zero illusion strength is zero illusion effect. A
    mirrored level appears twice, so each copy is given half its weight and the
    level contributes its information once, not twice.

    Weights are the inverse square of each level's standard error, so a level
    whose profile interval is wide pulls the curve less: an unidentified level
    carries a standard error an order of magnitude above its neighbours and is
    discounted without being dropped. The band is widened by the residual
    dispersion when that exceeds one and never narrowed below the standard
    errors, matching the rule for Figure 3.

    Args:
        levels: one row per |strength| with `x` in (0, 1], `y` the magnitude
                and `se` its standard error.
        df:     spline bases.
        n_grid: points across the full -1..1 axis.

    Returns:
        A frame with `x` in [-1, 1] and `fit`, `lo`, `hi`, or None when there
        are too few levels for the basis.
    """
    model = _fit_magnitude(levels, df)
    if model is None:
        return None
    design_info, beta, cov = model

    grid = np.linspace(-1.0, 1.0, n_grid)
    basis = _basis(design_info, grid)
    fit = basis @ beta
    var = np.einsum("ij,jk,ik->i", basis, cov, basis)
    half = _Z95 * np.sqrt(np.clip(var, 0.0, None))

    return pd.DataFrame({"x": grid, "fit": fit, "lo": fit - half, "hi": fit + half})


def mean_magnitude(
    levels: pd.DataFrame,
    df: int = MAGNITUDE_DF,
    n_grid: int = N_GRID,
) -> tuple[float, float] | None:
    """
    The smoothed magnitude averaged over strengths (0, 1], and its variance.

    Only the positive branch: the curve is odd by construction, so its mean
    over the full axis is zero. The mean is linear in the coefficients, so its
    variance is exact given their covariance.

    Returns:
        (mean, variance), or None when the curve cannot be fitted.
    """
    fit = _fit_magnitude(levels, df)
    if fit is None:
        return None
    design_info, beta, cov = fit

    grid = np.linspace(-1.0, 1.0, n_grid)
    weights = _basis(design_info, grid)[grid > 0].mean(0)
    return float(weights @ beta), float(weights @ cov @ weights)


def _fit_magnitude(levels: pd.DataFrame, df: int):
    """
    The weighted spline fit behind smooth_magnitude.

    Returns (design_info, beta, cov), with the dispersion rule already applied
    to `cov`, or None when there are too few levels for the basis.
    """
    sub = levels.dropna(subset=["x", "y", "se"]).sort_values("x")
    n = len(sub)
    if 2 * n + 1 <= df:
        return None

    xs = sub["x"].to_numpy(float)
    ys = sub["y"].to_numpy(float)
    ses = sub["se"].to_numpy(float)
    # A zero standard error would carry infinite weight; floor it relative to
    # the spread actually being fitted.
    ses = np.clip(ses, max(float(np.nanmedian(ses)) * 1e-3, 1e-12), None)

    x = np.concatenate([-xs, [0.0], xs])
    y = np.concatenate([-ys, [0.0], ys])
    se = np.concatenate([ses, [ses.min()], ses])
    share = np.concatenate([np.full(n, 0.5), [1.0], np.full(n, 0.5)])
    weights = share / se**2

    design = dmatrix(f"cr(x, df={df}) - 1", {"x": x}, return_type="dataframe")
    res = sm.WLS(y, design, weights=weights).fit()

    # statsmodels scales the covariance by the residual dispersion. Undo that
    # and reapply it floored at one, so the band only ever widens.
    cov = res.cov_params().to_numpy() / res.scale * max(float(res.scale), 1.0)
    return design.design_info, res.params.to_numpy(), cov
