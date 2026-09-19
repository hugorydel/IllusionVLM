"""
pipeline/human_comparison/robustness.py - Assumption checks on the Figure 2 fit.

The magnitude in Figure 2 comes from a joint fit over both illusion directions
at one |strength|, sharing a slope and a lapse pair and differing only in the
point of subjective equality:

    PSE(d) = side_bias - d * magnitude

Expressing the two directions' thresholds as a separation and a midpoint is an
exact reparameterisation, so no threshold information is lost. The substantive
assumption is the shared slope and lapses. This module tests it by likelihood
ratio at every level, against two alternatives:

    slope    sigma free per direction, lapses shared   (1 df)
    full     sigma and both lapses free per direction  (3 df)

The shared-slope model stays the reported estimator. It is the more constrained
of the two and is what makes a magnitude identifiable at levels where one
direction alone is too noisy to locate its own PSE; freeing the slope buys a
better fit at the cost of pushing more levels past identification. The point of
the test is to say where that constraint binds, and what it costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from pipeline.human_comparison.psychometrics import EPS, _grid_geometry, psychometric

_MODEL_SHAPES = {
    #            n_params, unpacker -> (mag, bias, s+, s-, lo+, hi+, lo-, hi-)
    "shared": (5, lambda p: (p[0], p[1], p[2], p[2], p[3], p[4], p[3], p[4])),
    "slope": (6, lambda p: (p[0], p[1], p[2], p[3], p[4], p[5], p[4], p[5])),
    "full": (8, lambda p: tuple(p)),
}


def _neg_log_lik(x, d, k, n, mag, bias, s_p, s_m, lo_p, hi_p, lo_m, hi_m) -> float:
    """Binomial negative log-likelihood allowing per-direction slope and lapses."""
    p = np.empty_like(x, dtype=float)
    for dv, s, lo, hi in ((+1.0, s_p, lo_p, hi_p), (-1.0, s_m, lo_m, hi_m)):
        mask = d == dv
        if mask.any():
            p[mask] = psychometric(x[mask], bias - dv * mag, s, lo, hi)
    p = np.clip(p, EPS, 1.0 - EPS)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))


def _fit(x, d, k, n, model: str) -> tuple[float, bool, tuple]:
    n_params, unpack = _MODEL_SHAPES[model]
    grid_max = float(np.max(np.abs(x)))
    _, _, rng = _grid_geometry(x)

    bias_b = (-1.5 * grid_max, 1.5 * grid_max)
    sigma_b = (rng / 500.0, rng * 3.0)
    lapse_b = (0.0, 0.45)

    # Same bounds and starts as fit_illusion_magnitude, extended per direction.
    bounds = [bias_b, bias_b, sigma_b] + (
        [sigma_b] if n_params > 5 else []
    ) + [lapse_b] * (n_params - (4 if n_params > 5 else 3))
    start = [0.1 * grid_max, 0.0, rng / 10.0] + (
        [rng / 10.0] if n_params > 5 else []
    ) + [0.02] * (n_params - (4 if n_params > 5 else 3))

    res = minimize(
        lambda p: _neg_log_lik(x, d, k, n, *unpack(p)),
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 20_000, "ftol": 1e-12},
    )
    return float(res.fun), bool(res.success), unpack(res.x)


def slope_asymmetry(sliced: pd.DataFrame) -> pd.DataFrame:
    """
    Likelihood-ratio test of the shared slope and lapses, one row per level.

    Args:
        sliced: cells carrying `direction` and `k_abs`, as
                `add_illusion_direction` returns them.

    Returns:
        One row per (species, illusion, k_abs) with the two p-values, the
        per-direction slopes under the relaxed model, and the magnitude under
        both models so the cost of the constraint is visible in delta units.
    """
    rows = []
    for (species, illusion, k_abs), grp in sliced[sliced["k_abs"] > 0].groupby(
        ["species", "illusion", "k_abs"]
    ):
        x = grp["true_diff"].to_numpy(float)
        d = grp["direction"].to_numpy(float)
        k = grp["n_positive"].to_numpy(float)
        n = grp["n_trials"].to_numpy(float)
        if len(np.unique(d)) < 2 or len(x) < 8:
            continue

        f_shared, ok_s, p_shared = _fit(x, d, k, n, "shared")
        f_slope, ok_l, p_slope = _fit(x, d, k, n, "slope")
        f_full, ok_f, _ = _fit(x, d, k, n, "full")

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "k_abs": k_abs,
                "p_slope": chi2.sf(max(2.0 * (f_shared - f_slope), 0.0), 1),
                "p_full": chi2.sf(max(2.0 * (f_shared - f_full), 0.0), 3),
                "sigma_dir_pos": round(p_slope[2], 5),
                "sigma_dir_neg": round(p_slope[3], 5),
                "magnitude_shared": round(p_shared[0], 5),
                "magnitude_slope_free": round(p_slope[0], 5),
                "converged": bool(ok_s and ok_l and ok_f),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Holm-Bonferroni across every level tested, since all are examined at once.
    out = out.sort_values("p_slope").reset_index(drop=True)
    m = len(out)
    thresholds = 0.05 / (m - np.arange(m))
    below = out["p_slope"].to_numpy() <= thresholds
    n_reject = int(np.argmin(below)) if not below.all() else m
    out["rejects_shared_slope"] = np.arange(m) < n_reject
    return out.sort_values(["species", "illusion", "k_abs"]).reset_index(drop=True)
