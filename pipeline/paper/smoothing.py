"""
pipeline/paper/smoothing.py - Smooth error rate against signed illusion strength.

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

    grid = np.linspace(-1.0, 1.0, n_grid)
    basis = build_design_matrices(
        [design.design_info], {"x": grid}, return_type="dataframe"
    )[0].to_numpy()

    beta = res.params.to_numpy()
    cov = res.cov_params().to_numpy()
    eta = basis @ beta
    # Diagonal of basis @ cov @ basis.T without forming the full matrix.
    var_eta = np.einsum("ij,jk,ik->i", basis, cov, basis) * dispersion
    half = _Z95 * np.sqrt(np.clip(var_eta, 0.0, None))

    return pd.DataFrame(
        {
            "x": grid,
            "fit": expit(eta) * 100.0,
            "lo": expit(eta - half) * 100.0,
            "hi": expit(eta + half) * 100.0,
        }
    )
